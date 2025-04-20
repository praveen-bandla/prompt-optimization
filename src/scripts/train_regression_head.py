"""
Module for training a regression head with LoRA and quantization.

Steps:
1. Load and preprocess the dataset.
2. Initialize the base model and tokenizer.
3. Apply LoRA for low-rank adaptation.
4. Quantize the model for efficiency.
5. Define the training loop with supervised fine-tuning.
6. Perform hyperparameter search for optimal performance.
7. Save the trained model and evaluation metrics.

Usage:
    PYTHONPATH=$(pwd)/../.. python train_regression_head.py      
"""
from configs.root_paths import *

import torch
import optuna
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorWithPadding
from peft import LoraConfig, get_peft_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load dataset
# Below is a dummy dataset
# BECCA: I know that you will move this to your section of setting up the classes/functions for handling this but I put it here for now to help me understand too
data = {
    "base_prompt": ["bp_1", "bp_2", "bp_3", "bp_4", "bp_5"],
    "prompt_variation": ["bpv_1", "bpv_2", "bpv_3", "bpv_4", "bpv_5"],
    "total_score": [18, 23, 20, 25, 0]
}
df = pd.DataFrame(data)
train_data, val_data = train_test_split(df, test_size=0.2)

# Tokenize function
def tokenize_function(batch, tokenizer, max_length=512):
    base_prompts = [item["base_prompt"] for item in batch]
    prompt_variations = [item["prompt_variation"] for item in batch]
    combined_text = [f"{bp} {pv}" for bp, pv in zip(base_prompts, prompt_variations)]

    encodings = tokenizer(
        combined_text,
        padding="longest",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

    # Add labels
    total_scores = [item["total_score"] for item in batch]
    encodings["labels"] = torch.tensor(total_scores, dtype=torch.float32).unsqueeze(1)

    return encodings

# Dataset class
class PromptDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "base_prompt": self.data.iloc[idx]["base_prompt"],
            "prompt_variation": self.data.iloc[idx]["prompt_variation"],
            "total_score": self.data.iloc[idx]["total_score"]
        }

train_dataset = PromptDataset(train_data)
val_dataset = PromptDataset(val_data)

# Collate function for dynamic tokenization
def collate_fn(batch, tokenizer):
    return tokenize_function(batch, tokenizer)

def objective(trial):
    # Hyperparameter search space
    lora_rank=trial.suggest_int("lora_rank", 4, 16)
    lora_alpha=trial.suggest_int("lora_alpha", 16, 64)
    dropout_rate=trial.suggest_uniform("dropout_rate", 0.1, 0.5)
    learning_rate=trial.suggest_loguniform("learning_rate", 1e-5, 1e-2)
    batch_size=trial.suggest_categorical("batch_size", [8, 16, 32])

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(PROMPT_OPT_BASE_MODEL_ID, use_safetensors=True) # must be base model
    tokenizer.pad_token = tokenizer.eos_token # Set pad token to eos token for causal LM
    # Alternatively, can use: tokenizer.add_special_tokenz({"pad_token": "[PAD]"})

    regression_head_model = AutoModelForCausalLM.from_pretrained(LORA_REGRESSION_HEAD_PATH, use_safetensors=True).to(device)

    # Apply LoRA
    lora_config = LoraConfig(
        r=lora_rank, # Lora attention dimension, the rank. Smaller r means fewer parameters are updated
        lora_alpha=lora_alpha, # Alpha scaling factor for LoRA
        target_modules=["q_proj", "v_proj"], 
        lora_dropout=dropout_rate # Help with regularization. Dropout probability for LoRA layers
    )
    model = get_peft_model(regression_head_model, lora_config)

    # BECCA: Skipping quantization step for now
    # Quantize model
    # model = torch.quantization.quantize_dynamic( # POTENTIAL ISSUE: Not all models/layers are compatible with dynamic quantization
    #     model, {torch.nn.Linear}, dtype=torch.qint8
    # )

    # Training setup
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, tokenizer)
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=lambda batch: collate_fn(batch, tokenizer)
    )

    # Optimizer and loss function
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.MSELoss()

    # Training loop
    for epoch in range(3):
        model.train()
        for batch in train_loader:
            inputs = {key: val.to(device) for key, val in batch.items() if key != "labels"}
            labels = batch["labels"].to(device)

            outputs = model(**inputs)
            logits = outputs.logits

            # PRAVEEN: I think this where the error is coming from. The logits and labels are not the same shape

            # PRAVEEN: fixing the shape issue
            logits = logits.mean(dim=1)

            # Debugging: Print shapes
            print(f'Epoch {epoch + 1}, Training - Logits shape: {logits.shape}, Labels shape: {labels.shape}')

            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                inputs = {key: val.to(device) for key, val in batch.items() if key != "labels"}
                labels = batch["labels"].to(device)

                outputs = model(**inputs)
                logits = outputs.logits

                # PRAVEEN: I think this where the error is coming from. The logits and labels are not the same shape

                # PRAVEEN: fixing the shape issue
                logits = logits.mean(dim=1)

                # Debugging: Print shapes
                print(f'Validation - Logits shape: {logits.shape}, Labels shape: {labels.shape}')

                val_loss = loss_fn(logits, labels)
                val_losses.append(val_loss.item())

        # Print validation loss
        print(f"Epoch {epoch + 1}, Validation Loss: {sum(val_losses) / len(val_losses)}")

    # Return the average validation loss
    return sum(val_losses) / len(val_losses)

# Run the hyperparameter optimiztion
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=10)

# Save model
# PRAVEEN: commenting out for now just to test the training code
# save_path = f"{LORA_REGRESSION_HEAD_PATH}_trial_{study.best_trial.number}"
# os.makedirs(save_path, exist_ok=True)
# model.save_pretrained(save_path)
# print(f"Model saved to {save_path}")