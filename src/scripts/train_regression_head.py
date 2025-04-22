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

# Dataset class
# class PromptDataset(torch.utils.data.Dataset):
#     def __init__(self, data):
#         self.data = data

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         return {
#             "base_prompt": self.data.iloc[idx]["base_prompt"],
#             "prompt_variation": self.data.iloc[idx]["prompt_variation"],
#             "total_score": self.data.iloc[idx]["total_score"]
#         }

# train_dataset = PromptDataset(train_data)
# val_dataset = PromptDataset(val_data)

# Collate function for dynamic tokenization
def collate_fn(batch, tokenizer):
    return tokenize_function(batch, tokenizer)

def load_models(lora_rank, lora_alpha, dropout_rate):
    tokenizer = AutoTokenizer.from_pretrained(PROMPT_OPT_BASE_MODEL_ID, use_safetensors=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(LORA_REGRESSION_HEAD_PATH, use_safetensors=True).to(device)

    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=dropout_rate,
        task_type=TaskType.SEQ_CLS
    )
    regression_head = get_peft_model(base_model, lora_config)

    return tokenizer, regression_head


# def compute_metrics(model, data_loader, loss_fn, tokenizer, split_name=""):
#     """
#     Computes RMSE and MAPE for a given dataset split (train, val, or test).

#     Args:
#         model: The trained model.
#         data_loader: DataLoader for the dataset split.
#         loss_fn: Loss function (e.g., MSELoss).
#         tokenizer: Tokenizer used for preprocessing.
#         split_name: Name of the dataset split (e.g., "train", "val", "test").

#     Returns:
#         dict: A dictionary containing RMSE and MAPE.
#     """
#     model.eval()
#     all_labels = []
#     all_predictions = []
#     losses = []

#     with torch.no_grad():
#         for batch in data_loader:
#             inputs = {key: val.to(device) for key, val in batch.items() if key != "labels"}
#             labels = batch["labels"].to(device)

#             outputs = model(**inputs)
#             logits = outputs.logits

#             # Adjust logits shape if necessary
#             logits = logits.mean(dim=1)

#             # Debugging: Print shapes
#             print(f"Logits shape: {logits.shape}, Labels shape: {labels.shape}")

#             # Compute loss
#             loss = loss_fn(logits, labels)
#             losses.append(loss.item())

#             # Collect predictions and labels for metrics
#             all_labels.extend(labels.cpu().numpy())
#             all_predictions.extend(logits.cpu().numpy())

#     # Compute metrics
#     #rmse = mean_squared_error(all_labels, all_predictions, squared=False)  # RMSE
#     mape = mean_absolute_percentage_error(all_labels, all_predictions)  # MAPE

#     print(f"{split_name} Metrics: RMSE={rmse:.4f}, MAPE={mape:.4f}")
#     return {"rmse": rmse, "mape": mape}


# def objective(trial):
#     # Hyperparameter search space
#     lora_rank = trial.suggest_int("lora_rank", 4, 16)
#     lora_alpha = trial.suggest_int("lora_alpha", 16, 64)
#     dropout_rate = trial.suggest_uniform("dropout_rate", 0.1, 0.5)
#     learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-2)
#     batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])

#     # Load model and tokenizer
#     tokenizer = AutoTokenizer.from_pretrained(PROMPT_OPT_BASE_MODEL_ID, use_safetensors=True)
#     tokenizer.pad_token = tokenizer.eos_token
#     tokenizer.pad_token_id = tokenizer.eos_token_id
#     tokenizer.padding_side = 'left'

#     regression_head_model = AutoModelForCausalLM.from_pretrained(LORA_REGRESSION_HEAD_PATH, use_safetensors=True).to(device)

#     # Apply LoRA
#     lora_config = LoraConfig(
#         r=lora_rank,
#         lora_alpha=lora_alpha,
#         target_modules=["q_proj", "v_proj"],
#         lora_dropout=dropout_rate,
#         task_type=TaskType.SEQ_CLS
#     )
#     model = get_peft_model(regression_head_model, lora_config)

#     # Training setup
#     train_loader = DataLoader(
#         train_data,
#         batch_size=batch_size,
#         shuffle=True,
#         collate_fn=lambda batch: collate_fn(batch, tokenizer)
#     )
#     val_loader = DataLoader(
#         val_data,
#         batch_size=batch_size,
#         collate_fn=lambda batch: collate_fn(batch, tokenizer)
#     )
#     test_loader = DataLoader(
#         test_data,
#         batch_size=batch_size,
#         collate_fn=lambda batch: collate_fn(batch, tokenizer)
#     )

#     # Optimizer and loss function
#     optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
#     loss_fn = torch.nn.MSELoss()

#     # Training loop
#     for epoch in range(3):
#         model.train()
#         for batch in train_loader:
#             inputs = {key: val.to(device) for key, val in batch.items() if key != "labels"}
#             labels = batch["labels"].to(device)

#             outputs = model(**inputs)
#             logits = outputs.logits

#             # Adjust logits shape if necessary
#             logits = logits.mean(dim=1)

#             # Debugging: Print shapes
#             print(f"Epoch {epoch + 1}, Training - Logits shape: {logits.shape}, Labels shape: {labels.shape}")


#             loss = loss_fn(logits, labels)
#             loss.backward()
#             optimizer.step()
#             optimizer.zero_grad()

#         # Validation metrics
#         print(f"Epoch {epoch + 1} Validation Metrics:")
#         compute_metrics(model, val_loader, loss_fn, tokenizer, split_name="Validation")

#     # Compute metrics on train, validation, and test datasets
#     print("Final Metrics:")
#     train_metrics = compute_metrics(model, train_loader, loss_fn, tokenizer, split_name="Train")
#     val_metrics = compute_metrics(model, val_loader, loss_fn, tokenizer, split_name="Validation")
#     test_metrics = compute_metrics(model, test_loader, loss_fn, tokenizer, split_name="Test")

#     # Return the average validation RMSE as the objective metric for Optuna
#     return model, tokenizer, val_metrics["rmse"]


# # Run the hyperparameter optimization
# best_model = None
# best_tokenizer = None

# def wrapped_objective(trial):
#     global best_model, best_tokenizer
#     model, tokenizer, val_loss = objective(trial)
#     best_model = model
#     best_tokenizer = tokenizer
#     return val_loss

# study = optuna.create_study(direction='minimize')
# study.optimize(wrapped_objective, n_trials=10)

# # Save the best model
# best_trial = study.best_trial
# print(f"Best trial: {best_trial.number}, Best trial params: {best_trial.params}")

# save_path = f"{LORA_REGRESSION_HEAD_PATH}_best_trial"
# os.makedirs(save_path, exist_ok=True)
# best_tokenizer.save_pretrained(save_path)
# best_model.save_pretrained(save_path)
# print(f"Trained regression head saved to {save_path}")


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

    regression_head_model = AutoModelForCausalLM.from_pretrained(LORA_REGRESSION_HEAD_PATH, use_safetensors=True).to(device)

    # Apply LoRA
    lora_config = LoraConfig(
        r=lora_rank, # Lora attention dimension, the rank. Smaller r means fewer parameters are updated
        lora_alpha=lora_alpha, # Alpha scaling factor for LoRA
        target_modules=["q_proj", "v_proj"], 
        lora_dropout=dropout_rate, # Help with regularization. Dropout probability for LoRA layers
        task_type = TaskType.SEQ_CLS # Sequence classification task
    )
    model = get_peft_model(regression_head_model, lora_config)

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
        collate_fn=lambda batch: collate_fn(batch, tokenizer)
    )
    val_loader = DataLoader(
        val_data,
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