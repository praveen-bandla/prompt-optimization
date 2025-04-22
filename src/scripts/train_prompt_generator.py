"""
Module for training the prompt generator with LoRA SFT.

Steps:


Usage:
    PYTHONPATH=$(pwd)/../.. python train_regression_head.py      
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

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_dataset():
    # BECCA: This would need to be moved into the base prompt class or something
    # Need to put SQLite database into a list of dictionaries supposedly
    # This is the format that leo suggested
    data = [
        {"base_prompts": "Write a summary of the article."},
        {"base_prompts": "Explain the concept of gravity."},
        {"base_prompts": "Describe the process of photosynthesis."}
    ]

    dataset = Dataset.from_list(data)

    return dataset

def load_models(lora_rank, lora_alpha, dropout_rate):
    """Load the regression head and prompt generator models."""
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(PROMPT_OPT_BASE_MODEL_ID, use_safetensors=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'left' # Required for decoder-only models like Llama

    # Load regression head
    regression_head = AutoModelForSequenceClassification.from_pretrained(
        LORA_REGRESSION_HEAD_PATH, 
        use_safetensors=True).to(device)
    # Explicitly set pad_token_id for config
    regression_head.config.pad_token_id = tokenizer.pad_token_id
    regression_head.eval()

    # Load prompt generator
    prompt_generator = AutoModelForCausalLM.from_pretrained(LORA_PROMPT_GEN_PATH, use_safetensors=True).to(device)
    prompt_generator = apply_lora(prompt_generator, lora_rank, lora_alpha, dropout_rate)
    prompt_generator.train()

    return tokenizer, regression_head, prompt_generator

def format_prompt_with_instruction(base_prompt):
    """Format the base prompt with an instruction."""
    # BECCA: I know this is hardcoded for now, placeholder for efficiency
    instruction = "Instruction: Rewrite the prompt to improve LLM output quality while preserving meaning."
    return f"[INST] {instruction}\nBase Prompt: {base_prompt} [\\INST]"

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
    loss_fn = torch.nn.CrossEntropyLoss()

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True)
        # Q: do we need a collate_fn? answer is no

    for epoch in range(3):
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            base_prompts = batch["base_prompts"]

            # Format base prompts with instructions
            formatted_prompts = [format_prompt_with_instruction(prompt) for prompt in base_prompts]

            # Generate prompt variations
            inputs = tokenizer(formatted_prompts, return_tensors="pt", padding=True, truncation=True).to(device)
            outputs = prompt_generator.generate(**inputs, num_return_sequences=3, max_length=50) # eventually want to remove hard-coding num_return_sequences
            variations = tokenizer.batch_decode(outputs, skip_special_tokens=False)

            # Score variations using regression head
            variation_inputs = tokenizer(variations, return_tensors="pt", padding=True, truncation=True).to(device)
            scores = regression_head(
                input_ids = variation_inputs['input_ids'],
                attention_mask = variation_inputs['attention_mask']
                ).logits

            # Compute softmax weights for the scores
            weights = softmax(scores, dim=1)
            # Might need dim = 0 if scores is a single-dimensional tensor

            # Compute weighted loss
            weighted_loss = torch.sum(weights * scores) / scores.size(0) # Average loss across batch

            # Backpropagation
            optimizer.zero_grad()
            weighted_loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1} completed. Loss: {weighted_loss.item()}")
    
    return weighted_loss.item()

def objective(trial, train_dataset):
    """Optuna objective function for hyperparameter tuning."""
    return train_prompt_generator(trial, train_dataset)

if __name__ == "__main__":
    train_dataset = load_dataset()
    # BECCA: When implemented for real, should be something like
    # train_dataset = load_dataset_from_sqlite(dp_bath, table_name)

    # Hyperparameter tuning with Optuna
    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda trial: objective(trial, train_dataset), 
        n_trials=5)

    # Save the best model
    best_trial = study.best_trial
    print(f"Best trial: {best_trial.params}")

    # Reload tokenizer and prompt generator for saving
    save_path = f"{LORA_PROMPT_GEN_PATH}_trial_{best_trial.number}"
    os.makedirs(save_path, exist_ok=True)
    tokenizer, _, prompt_generator = load_models(
        best_trial.params["lora_rank"],
        best_trial.params["lora_alpha"],
        best_trial.params["dropout_rate"]
    )
    tokenizer.save_pretrained(save_path)
    prompt_generator.save_pretrained(save_path)
    print(f"Trained prompt generator saved to {save_path}")