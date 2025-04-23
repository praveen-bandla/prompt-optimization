"""
Module for training a regression head with LoRA.

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
from src.utils.data_handler import RegressionHeadDataset

import torch
import optuna
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorWithPadding
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error


device = "cuda" if torch.cuda.is_available() else "cpu"

# Load dataset
# Below is a dummy dataset
# BECCA: I know that you will move this to your section of setting up the classes/functions for handling this but I put it here for now to help me understand too
# data = {
#     "base_prompt": ["bp_1", "bp_2", "bp_3", "bp_4", "bp_5"],
#     "prompt_variation": ["bpv_1", "bpv_2", "bpv_3", "bpv_4", "bpv_5"],
#     "total_score": [18, 23, 20, 25, 0]
# }
# df = pd.DataFrame(data)
# train_data, val_data = train_test_split(df, test_size=0.2)

dataset = RegressionHeadDataset()
train_data, val_data, test_data = dataset.get_data_split()

# Tokenize function
def tokenize_function(batch, tokenizer, max_length=512):
    # base_prompts = [item["base_prompt"] for item in batch]
    # prompt_variations = [item["prompt_variation"] for item in batch]
    base_prompts = [item[0] for item in batch]  # Extract base_prompt
    prompt_variations = [item[1] for item in batch]  # Extract prompt_variation
    combined_text = [f"{bp} {pv}" for bp, pv in zip(base_prompts, prompt_variations)]

    encodings = tokenizer(
        combined_text,
        padding="longest",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

    # Add labels
    #total_scores = [item["total_score"] for item in batch]
    total_scores = [item[2] for item in batch]  # Extract total_score
    encodings["labels"] = torch.tensor(total_scores, dtype=torch.float32).unsqueeze(1)

    return encodings

def objective(trial):
    # Hyperparameter search space
    lora_rank=trial.suggest_int("lora_rank", 4, 16)
    lora_alpha=trial.suggest_int("lora_alpha", 16, 64)
    dropout_rate=trial.suggest_uniform("dropout_rate", 0.1, 0.5)
    learning_rate=trial.suggest_loguniform("learning_rate", 1e-5, 1e-2)
    batch_size=trial.suggest_categorical("batch_size", [8, 16, 32])

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(PROMPT_OPT_BASE_MODEL_ID, use_safetensors=True) # must be base model
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id # Set pad token to eos token for causal LM
    # Alternatively, can use: tokenizer.add_special_tokenz({"pad_token": "[PAD]"})
    tokenizer.padding_side = 'left'

    regression_head_model = AutoModelForCausalLM.from_pretrained(
        LORA_REGRESSION_HEAD_PATH, 
        use_safetensors=True,
        output_hidden_states=True
        ).to(device)

    # Apply LoRA
    lora_config = LoraConfig(
        r=lora_rank, # Lora attention dimension, the rank. Smaller r means fewer parameters are updated
        lora_alpha=lora_alpha, # Alpha scaling factor for LoRA
        target_modules=["q_proj", "v_proj"], 
        lora_dropout=dropout_rate, # Help with regularization. Dropout probability for LoRA layers
        task_type = TaskType.FEATURE_EXTRACTION
    )
    model = get_peft_model(regression_head_model, lora_config)

    model.regression_head = nn.Linear(model.config.hidden_size, 1).to(device)

    # BECCA: Skipping quantization step for now
    # Quantize model
    # model = torch.quantization.quantize_dynamic( # POTENTIAL ISSUE: Not all models/layers are compatible with dynamic quantization
    #     model, {torch.nn.Linear}, dtype=torch.qint8
    # )

    # Training setup
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: tokenize_function(batch, tokenizer)
    )
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        collate_fn=lambda batch: tokenize_function(batch, tokenizer)
    )
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        collate_fn=lambda batch: tokenize_function(batch, tokenizer)
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
            # Extract hidden states and pass through regression head
            hidden_states = outputs.hidden_states[-1]  # Shape: [batch_size, sequence_length, hidden_size]
            pooled_output = hidden_states[:, 0, :]  # Use the first token's hidden state
            logits = model.regression_head(pooled_output)  # Shape: [batch_size, 1]

            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Validation
        model.eval()
        val_losses = []
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                inputs = {key: val.to(device) for key, val in batch.items() if key != "labels"}
                labels = batch["labels"].to(device)

                outputs = model(**inputs)


                # PRAVEEN: Commenting this out for now, this is where the bug is
                # logits = outputs.logits
                # logits = logits.mean(dim=1)

                # Extract hidden states
                hidden_states = outputs.hidden_states[-1]  # Shape: [batch_size, sequence_length, hidden_size]

                # Use the first token's hidden state (e.g., [CLS] token) as the pooled representation
                pooled_output = hidden_states[:, 0, :]  # Shape: [batch_size, hidden_size]

                # Pass through the regression head
                logits = model.regression_head(pooled_output)  # Shape: [batch_size, 1]

                val_loss = loss_fn(logits, labels)
                val_losses.append(val_loss.item())
                
                # Collect predictions and labels for metrics
                all_preds.extend(logits.squeeze(-1).cpu().numpy())  # Ensure 1D array
                all_labels.extend(labels.squeeze(-1).cpu().numpy())  # Ensure 1D array

                # val_loss = loss_fn(logits, labels)
                # val_losses.append(val_loss.item())

        # compute metrics
        rmse = mean_squared_error(all_labels, all_preds, squared=False)
        mape = mean_absolute_percentage_error(all_labels, all_preds)


        # Print validation loss
        print(f"Epoch {epoch + 1}, Validation Loss: {sum(val_losses) / len(val_losses)}")
        print(f"Epoch {epoch + 1}, RMSE: {rmse}")
        print(f"Epoch {epoch + 1}, MAPE: {mape}")

    # Test Evaluation
    model.eval()
    test_losses = []
    test_preds = []
    test_labels = []

    with torch.no_grad():
        for batch in test_loader:
            inputs = {key: val.to(device) for key, val in batch.items() if key != "labels"}
            labels = batch["labels"].to(device)

            outputs = model(**inputs)

            # Extract hidden states and pass through regression head
            hidden_states = outputs.hidden_states[-1]  # Get the last layer's hidden states
            pooled_output = hidden_states[:, 0, :]  # Use the first token's hidden state
            logits = model.regression_head(pooled_output)  # Shape: [batch_size, 1]

            # Compute test loss
            test_loss = loss_fn(logits, labels)
            test_losses.append(test_loss.item())

            # Collect predictions and labels for metrics
            test_preds.extend(logits.squeeze(-1).cpu().numpy())  # Ensure 1D array
            test_labels.extend(labels.squeeze(-1).cpu().numpy())  # Ensure 1D array

    # Compute RMSE and MAPE for the test set
    test_rmse = mean_squared_error(test_labels, test_preds, squared=False)  # RMSE
    test_mape = mean_absolute_percentage_error(test_labels, test_preds)  # MAPE

    # Print test results
    print(f"Test Loss: {sum(test_losses) / len(test_losses)}")
    print(f"Test RMSE: {test_rmse}")
    print(f"Test MAPE: {test_mape}")

    # Return the average validation loss
    return model, tokenizer, sum(val_losses) / len(val_losses)

# Run the hyperparameter optimiztion
best_model = None
best_tokenizer = None

def wrapped_objective(trial):
    global best_model, best_tokenizer
    model, tokenizer, val_loss = objective(trial)
    best_model = model
    best_tokenizer = tokenizer
    return val_loss

study = optuna.create_study(direction='minimize')
study.optimize(wrapped_objective, n_trials=10)


# Save model
# PRAVEEN: commenting out for now just to test the training code
# Save the best model
best_trial = study.best_trial
print(f"Best trial: {best_trial.number}, Best trials params: {best_trial.params}")

# Reload tokenizer and prompt generator for saving
save_path = f"{LORA_REGRESSION_HEAD_PATH}_best_trial"
os.makedirs(save_path, exist_ok=True)
best_tokenizer.save_pretrained(save_path)
best_model.save_pretrained(save_path)
print(f"Trained regression head saved to {save_path}")

# BECCA: Old save code below
# save_path = f"{LORA_REGRESSION_HEAD_PATH}_trial_{study.best_trial.number}"

# os.makedirs(save_path, exist_ok=True)
# model.save_pretrained(save_path)
# print(f"Model saved to {save_path}")