'''
Here is the code for training the prompt generator using the reward system
'''


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
    """
    Load the dataset containing base prompts from a database or file.
    Returns:
        Dataset: A HuggingFace Dataset object containing base prompts.
    """
    data = [
        {"bp_idx": 0, "base_prompt_string": "Write a summary of the article."},
        {"bp_idx": 1, "base_prompt_string": "Explain the concept of gravity."},
        {"bp_idx": 2, "base_prompt_string": "Describe the process of photosynthesis."}
    ]

    dataset = Dataset.from_list(data)

    return dataset

def format_prompt_with_instruction(base_prompt):
    """
    Format the base prompt with an instruction and an example to guide the model in generating a prompt variation.
    Args:
        base_prompt (str): The base prompt to be formatted.
    Returns:
        str: A single formatted input string for the model.
    """
    # Example of the desired behavior
    example_base_prompt = "Create a learning guide for a third-grader on the concept of gravity."
    example_variation = "Draft a concise learning guide for a third-grader on gravity. Make it fun and engaging with strong structure."

    # Instruction with an example
    instruction = (
        f"Rewrite the following base instruction to create an alternative LLM prompt phrasing. Adjust the wording and structure of the prompt and include cues to create guidance for a model to generate a higher quality response."
        f"The rewritten prompt MUST include the EXACT phrase 'learning guide' and the word 'third-grade' or 'third-grader'. "
        f"Do not include any additional text or explanation. Return only the rewritten prompt.\n\n"
        f"Example:\n"
        f"Base Instruction: {example_base_prompt}\n"
        f"Rewritten Prompt: {example_variation}\n\n"
        f"Now, rewrite the following base instruction:\n"
        f"Base Instruction: {base_prompt}\n"
        f"Rewritten Prompt:"
    )

    return instruction

def load_prompt_gen_model(lora_rank, lora_alpha, dropout_rate):
    """
    Load the base prompt generator model to be fine-tuned with LoRA. Also load the tokenizer.
    Args:
        lora_rank (int): The rank for LoRA.
        lora_alpha (int): The alpha parameter for LoRA.
        dropout_rate (float): The dropout rate for LoRA.
    Returns:
        AutoTokenizer: The tokenizer for the model.
        AutoModelForCausalLM: The prompt generator model with LoRA applied.
    """
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(PROMPT_GEN_BASE_MODEL_ID, use_safetensors=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'left'  # Required for decoder-only models like LLaMA

    # Load the base prompt generator model
    prompt_generator = AutoModelForCausalLM.from_pretrained(
        LORA_PROMPT_GEN_PATH,
        use_safetensors=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # Apply LoRA for fine-tuning
    lora_config = LoraConfig(
        r=lora_rank,  # LoRA attention dimension (rank)
        lora_alpha=lora_alpha,  # Alpha scaling factor for LoRA
        target_modules=["q_proj", "v_proj"],  # Target modules for LoRA
        lora_dropout=dropout_rate,  # Dropout probability for LoRA layers
        task_type=TaskType.CAUSAL_LM  # Task type for causal language modeling
    )
    prompt_generator = get_peft_model(prompt_generator, lora_config)

    # Set the model to training mode
    prompt_generator.train()

    print("Prompt generator model and tokenizer loaded successfully.")
    return tokenizer, prompt_generator


def generate_prompt_variation(prompt_generator, tokenizer, formatted_prompt, max_new_tokens=50):
    """
    Generate a prompt variation using the finetuned prompt generator model.
    Args:
        prompt_generator (AutoModelForCausalLM): The finetuned prompt generator model.
        tokenizer (AutoTokenizer): The tokenizer used for the model.
        formatted_prompt (str): The formatted input string.
        max_new_tokens (int): The maximum number of new tokens to generate.
    Returns:
        str: The generated prompt variation.
    """
    # Tokenize the input
    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    ).to(device)

    # Generate output
    outputs = prompt_generator.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )

    # Decode the generated tokens
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return generated_text

def parse_model_output(generated_text):
    """
    Parse the output from the model to extract the generated prompt variation.
    Args:
        generated_text (str): The output text from the model.
    Returns:
        str: The generated prompt variation.
    """
    # Extract the generated prompt variation from the model output
    # Assuming the output format is consistent and we can split by newlines
    lines = generated_text.split("\n")
    for line in lines:
        if line.startswith("Rewritten Prompt:"):
            print("Found rewritten prompt line:", line)
            return line.replace("Rewritten Prompt:", "").strip()

def load_regression_head_model():
    '''
    Load the regression head model for scoring the generated prompt variations.
    '''
    # Load the regression head model
    regression_head_model = AutoModelForCausalLM.from_pretrained(
        BEST_LORA_REGRESSION_HEAD_PATH
    ).to(device)

    regression_head_model.config.output_hidden_states = True
    tokenizer = AutoTokenizer.from_pretrained(
        REGRESSION_HEAD_BASE_MODEL_ID,
        use_safetensors=True
    )

    return regression_head_model, tokenizer


def call_regression_head(regression_head_model, tokenizer, base_prompt, prompt_variation):
    '''
    Call the regression head model to get the score for the generated prompt variation.
    Args:
        regression_head_model (AutoModelForCausalLM): The regression head model.
        tokenizer (AutoTokenizer): The tokenizer for the regression head model.
        base_prompt (str): The base prompt.
        prompt_variation (str): The generated prompt variation.
    Returns:
        float: The score for the generated prompt variation.
    '''
    combined_input = f'{base_prompt} {prompt_variation}'

    inputs = tokenizer(
        combined_input,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    ).to(device)

    # Get the model's output
    with torch.no_grad():
        outputs = regression_head_model(**inputs)
        hidden_states = outputs.hidden_states[-1]
        pooled_output = hidden_states.mean(dim=1).to(torch.float16).to(device)
        logits = regression_head_model.regression_head(pooled_output)
        prediction = logits.squeeze(-1).cpu().item()

    return prediction

def retrieve_base_prompt_score(base_prompt_id):
    """
    Retrieve the score of the base prompt from a stored validation dataset.
    Args:
        base_prompt_id (int): The ID of the base prompt.
    Returns:
        float: The score of the base prompt.
    """
    pass

def calculate_reward(base_score, variation_score):
    """
    Calculate the reward based on the difference between the base prompt score and the variation score.
    Args:
        base_score (float): The score of the base prompt.
        variation_score (float): The score of the prompt variation.
    Returns:
        float: The calculated reward.
    """
    pass

def apply_lora(model, lora_rank, lora_alpha, dropout_rate):
    """
    Apply LoRA configuration to the model for efficient fine-tuning.
    Args:
        model (torch.nn.Module): The model to apply LoRA to.
        lora_rank (int): The rank for LoRA.
        lora_alpha (int): The alpha parameter for LoRA.
        dropout_rate (float): The dropout rate for LoRA.
    Returns:
        torch.nn.Module: The model with LoRA applied.
    """
    pass

def train_prompt_generator(trial, train_dataset):
    """
    Train the prompt generator using reward-based training.
    Args:
        trial (optuna.Trial): The Optuna trial object for hyperparameter tuning.
        train_dataset (Dataset): The training dataset.
    Returns:
        torch.nn.Module: The trained prompt generator model.
        AutoTokenizer: The tokenizer used for the model.
        float: The final loss value.
    """
    pass

def objective(trial, train_dataset):
    """
    Optuna objective function for hyperparameter tuning.
    Args:
        trial (optuna.Trial): The Optuna trial object.
        train_dataset (Dataset): The training dataset.
    Returns:
        float: The loss value for the trial.
    """
    pass

def save_model_and_tokenizer(model, tokenizer, save_path):
    """
    Save the trained model and tokenizer to the specified path.
    Args:
        model (torch.nn.Module): The trained model.
        tokenizer (AutoTokenizer): The tokenizer used for the model.
        save_path (str): The directory path to save the model and tokenizer.
    """
    pass

def main():
    """
    Main function to load the dataset, perform hyperparameter tuning, and train the model.
    """
    pass


if __name__ == "__main__":
    # Define LoRA parameters
    lora_rank = 8
    lora_alpha = 32
    dropout_rate = 0.1

    # Load models
    print("Loading models...")
    tokenizer, prompt_generator = load_prompt_gen_model(lora_rank, lora_alpha, dropout_rate)
    print("Models loaded successfully.")

    # Define a test base prompt
    test_base_prompt = "Make a learning guide for a third-grader on the concept of fractions."

    print(f'Base prompt: {test_base_prompt}')
    formatted_prompt = format_prompt_with_instruction(test_base_prompt)

    # Run inference to generate a prompt variation
    print("\nGenerating prompt variation...")
    generated_variation = generate_prompt_variation(prompt_generator, tokenizer, formatted_prompt)
    parsed_variation = parse_model_output(generated_variation)
    print("Generated prompt variation:")
    print(parsed_variation)

    # Load regression head model
    regression_head_model, regression_head_tokenizer = load_regression_head_model()

    print("Regression head model loaded successfully.")

    # Call regression head model to get the score for the generated prompt variation
    print("\nCalling regression head model...")
    pv_score = call_regression_head(regression_head_model, regression_head_tokenizer, test_base_prompt, parsed_variation)
    print("Prompt variation score:", pv_score)

