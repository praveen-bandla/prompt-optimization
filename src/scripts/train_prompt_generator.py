"""
Module for training the prompt generator with LoRA SFT.

Steps:


Usage:
    PYTHONPATH=$(pwd)/../.. python train_prompt_generator.py      
"""
from configs.root_paths import *

import torch
import optuna
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from peft import get_peft_model, LoraConfig, TaskType
from torch.utils.data import DataLoader
from torch.nn.functional import softmax
from datasets import Dataset
from tqdm import tqdm
import os
import pandas as pd
import json

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_dataset():
    # BECCA: This would need to be moved into the base prompt class or something
    # Need to put SQLite database into a list of dictionaries supposedly
    # This is the format that leo suggested
    data = [
        {"bp_idx": 0, "base_prompt_string": "Write a summary of the article."},
        {"bp_idx": 1, "base_prompt_string": "Explain the concept of gravity."},
        {"bp_idx": 2, "base_prompt_string": "Describe the process of photosynthesis."}
    ]

    dataset = Dataset.from_list(data)

    return dataset

def load_models(lora_rank, lora_alpha, dropout_rate):
    """Load the regression head and prompt generator models."""
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(PROMPT_GEN_BASE_MODEL_ID, use_safetensors=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'left' # Required for decoder-only models like Llama

    # Load regression head
    regression_head = AutoModelForSequenceClassification.from_pretrained(
        BEST_LORA_REGRESSION_HEAD_PATH, 
        use_safetensors=True,
        num_labels = 1
    ).to(device)
    # Explicitly set pad_token_id for config
    regression_head.config.pad_token_id = tokenizer.pad_token_id
    regression_head.eval()

    # Ensure regression_head parameters require gradients
    for param in regression_head.parameters():
        param.requires_grad = True

    # Load prompt generator
    prompt_generator = AutoModelForCausalLM.from_pretrained(LORA_PROMPT_GEN_PATH, use_safetensors=True).to(device)
    prompt_generator = apply_lora(prompt_generator, lora_rank, lora_alpha, dropout_rate)
    prompt_generator.train()

    return tokenizer, regression_head, prompt_generator

def format_prompt_with_instruction(base_prompt):
    """Format the base prompt with an instruction."""
    # BECCA: I know this is hardcoded for now, placeholder for efficiency
    # instruction = "Instruction: Rewrite the prompt to improve LLM output quality while preserving meaning."

    # instruction = "You are an expert in prompt engineering. Your task is to write one alternative LLM prompt phrasing for the base instruction: '{base_prompt}'. Your variation should instruct a language model (not a child) to generate learning guides for third-grade students. Return exactly one prompt variation for the base prompt input. The prompt must be a instruction to an LLM for generating a learning guide for third-grade students. You must include the exact word “guide”, and include either “third-grade” or “third-grader”. Do **not** use synonyms like 'manual', 'walkthrough', 'tutorial', or 'primer'. The prompt must contain the literal word “guide”. IMPORTANT: Print only the prompt variation in the format:"

    # messages = [
    # {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    # {"role": "user", "content": "Who are you?"},
    # ]
    if not os.path.exists(PROMPT_GENERATOR_MODEL_INPUT):
        raise FileNotFoundError(f"Instruction file not found at {PROMPT_GENERATOR_MODEL_INPUT}")
    with open(PROMPT_GENERATOR_MODEL_INPUT, 'r') as f:
        prompt_structure = json.load(f)

    system_role = prompt_structure["system_role"]
    content = prompt_structure["content_template"]

    system_role = system_role.format(base_prompt=base_prompt)

    full_prompt = [
        {
            "role": "system",
            "content": system_role
        },
        {
            "role": "user",
            "content": content
        }
    ]
    print(f"Full prompt: {full_prompt}")
    return full_prompt


def apply_lora(model, lora_rank, lora_alpha, dropout_rate):
    """Apply LoRA configuration to the model."""
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=dropout_rate,
        bias="none", # biases aren't updated
        task_type=TaskType.CAUSAL_LM, # Task is causal language modeling
    )
    return get_peft_model(model, lora_config)

best_model = None
best_tokenizer = None
best_loss = None

def train_prompt_generator(trial, train_dataset):
    """Train the prompt generator using the regression head's feedback."""
    # Hyperparameters
    lora_rank = trial.suggest_int("lora_rank", 4, 16)
    lora_alpha = trial.suggest_int("lora_alpha", 16, 64)
    dropout_rate = trial.suggest_uniform("dropout_rate", 0.1, 0.5)
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])

    tokenizer, regression_head, prompt_generator = load_models(lora_rank, lora_alpha, dropout_rate)

    optimizer = torch.optim.AdamW(prompt_generator.parameters(), lr=learning_rate)
    # loss_fn = torch.nn.CrossEntropyLoss()

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True)
        # Q: do we need a collate_fn? answer is no

    for epoch in range(3):
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            base_prompts = batch["base_prompt_string"]
            bp_indices = batch["bp_idx"]

            for bp_idx, base_prompt in zip(bp_indices, base_prompts):
                # Format base prompts with instructions
                formatted_prompts = format_prompt_with_instruction(base_prompt)

                # Generate prompt variations
                input_tensor = tokenizer.apply_chat_template(
                    formatted_prompts, 
                    add_generation_prompt=True, 
                    return_tensors='pt'
                ).to(device)

                # Extract input_ids and attention_mask for the generate method
                input_dict = {
                    "input_ids": input_tensor["input_ids"],
                    "attention_mask": input_tensor["attention_mask"]
                }

                # Generate one prompt variation
                outputs = prompt_generator.generate(
                    **input_dict, 
                    num_return_sequences=1, 
                    max_new_tokens=50
                )

                # Decode the single variation
                variations = tokenizer.batch_decode(outputs, skip_special_tokens=False)

                # Score the variation using the regression head
                variation_inputs = tokenizer(
                    variations, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True
                ).to(device)
                scores = regression_head(
                    input_ids=variation_inputs['input_ids'],
                    attention_mask=variation_inputs['attention_mask']
                ).logits.squeeze(dim=1)  # Shape: (1,)

                # Ensure correct number of scores is produced
                if scores.shape[0] != 1:
                    raise ValueError(f"Expected 1 score, but got {scores.shape[0]} for bp_idx {bp_idx}")

                # Load total_score from the parquet file
                filepath = f'{VALIDATION_SCORES}/{bp_idx}_validation_score.parquet'
                validation_scores = pd.read_parquet(filepath)
                total_score = validation_scores.loc[
                    validation_scores['bpv_idx'].apply(lambda x: list(x) == [bp_idx, -1]), 'total_score'].values[0]

                # Calculate the reward for the variation
                reward = scores.item() - total_score

                # Maximize the reward
                loss = -reward

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Output check
                print(f"bp_idx: {bp_idx}, Base Prompt: {base_prompt}, Variation: {variations[0]}")
                print(f"bp_idx: {bp_idx}, Score: {scores.item()}, Validator Score: {total_score}")
                print(f"bp_idx: {bp_idx}, Loss: {loss.item()}, Reward: {reward}")


        print(f"Epoch {epoch + 1} completed.")
    
    return prompt_generator, tokenizer, loss.item()

def objective(trial, train_dataset):
    """Optuna objective function for hyperparameter tuning."""
    global best_model, best_tokenizer, best_loss
    model, tokenizer, loss = train_prompt_generator(trial, train_dataset)

    if best_loss is None or loss > best_loss:
        best_loss = loss
        best_model = model
        best_tokenizer = tokenizer

        # Save the best model and tokenizer
        save_path = BEST_LORA_PROMPT_GEN_PATH
        os.makedirs(save_path, exist_ok=True)
        tokenizer.save_pretrained(save_path)
        model.save_pretrained(save_path)
        print(f"New best model saved with loss: {loss}")

    return loss

if __name__ == "__main__":
    train_dataset = load_dataset()
    # BECCA: When implemented for real, should be something like
    # train_dataset = load_dataset_from_sqlite(dp_bath, table_name)

    # Hyperparameter tuning with Optuna
    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda trial: objective(trial, train_dataset), 
        n_trials=10)

    # Save the best model
    best_trial = study.best_trial
    print(f"Best trial: {best_trial.number}, Best trials params: {best_trial.params}")

    # # Reload tokenizer and prompt generator for saving
    save_path = BEST_LORA_PROMPT_GEN_PATH
    # os.makedirs(save_path, exist_ok=True)
    # best_tokenizer.save_pretrained(save_path)
    # best_model.save_pretrained(save_path)
    print(f"Trained prompt generator saved to {save_path}")