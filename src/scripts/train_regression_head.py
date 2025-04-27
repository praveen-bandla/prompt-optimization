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
import json

device = "cuda" if torch.cuda.is_available() else "cpu"

def prepare_metrics_for_json(metrics):
    """
    Prepare the metrics dictionary for JSON serialization by converting all numerical values to float.
    Args:
        metrics (dict): The metrics dictionary to prepare.
    Returns:
        dict: A JSON-serializable dictionary.
    """
    prepared_metrics = {}
    for key, value in metrics.items():
        if isinstance(value, list):
            # Convert all elements in the list to float
            prepared_metrics[key] = [float(v) for v in value]
        elif isinstance(value, (int, float)):  # Handle scalar values
            prepared_metrics[key] = float(value)
        else:
            prepared_metrics[key] = value  # Leave non-numerical values as-is
    return prepared_metrics

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
    encodings["labels"] = torch.tensor(total_scores, dtype=torch.float16).unsqueeze(1)

    return encodings

def objective(trial):
    # Hyperparameter search space
    lora_rank=trial.suggest_int("lora_rank", 4, 16)
    lora_alpha=trial.suggest_int("lora_alpha", 16, 64)
    dropout_rate=trial.suggest_uniform("dropout_rate", 0.1, 0.5)
    learning_rate=trial.suggest_loguniform("learning_rate", 1e-5, 1e-2)
    batch_size=trial.suggest_categorical("batch_size", [8, 16, 32])

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(REGRESSION_HEAD_BASE_MODEL_ID, use_safetensors=True) # must be base model
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id # Set pad token to eos token for causal LM
    # Alternatively, can use: tokenizer.add_special_tokenz({"pad_token": "[PAD]"})
    tokenizer.padding_side = 'left'

    regression_head_model = AutoModelForCausalLM.from_pretrained(
        LORA_REGRESSION_HEAD_PATH, 
        use_safetensors=True,
        output_hidden_states=True,
        return_dict_in_generate =True,
        torch_dtype=torch.float16,
        device_map="auto"
        )
    #regression_head_model.config.return_dict_in_generate = True

    # Apply LoRA
    lora_config = LoraConfig(
        r=lora_rank, # Lora attention dimension, the rank. Smaller r means fewer parameters are updated
        lora_alpha=lora_alpha, # Alpha scaling factor for LoRA
        target_modules=["q_proj", "v_proj"], 
        lora_dropout=dropout_rate, # Help with regularization. Dropout probability for LoRA layers
        task_type = TaskType.FEATURE_EXTRACTION
    )
    model = get_peft_model(regression_head_model, lora_config)

    model.regression_head = nn.Linear(model.config.hidden_size, 1).to(device).to(torch.float16)

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

    metrics = {
        "train_rmse": [],
        "val_rmse": [],
        "test_rmse": [],
        "train_loss": [],
        "val_loss": [],
        "test_loss": []
    }

    model.eval()  # Set model to evaluation mode

    # Function to compute RMSE for a given data loader
    def compute_rmse(loader):
        preds = []
        labels = []
        with torch.no_grad():
            for batch in loader:
                inputs = {key: val.to(device) for key, val in batch.items() if key != "labels"}
                labels_batch = batch["labels"].to(device).to(torch.float16)

                outputs = model(**inputs)
                hidden_states = outputs.hidden_states[-1]  # Shape: [batch_size, sequence_length, hidden_size]
                pooled_output = hidden_states.mean(dim=1).to(device)  # Shape: [batch_size, hidden_size]
                logits = model.regression_head(pooled_output)  # Shape: [batch_size, 1]

                preds.extend(logits.squeeze(-1).detach().cpu().numpy())
                labels.extend(labels_batch.squeeze(-1).detach().cpu().numpy())
        rmse = mean_squared_error(labels, preds, squared=False)
        return rmse

    # Compute RMSE for train, validation, and test datasets
    initial_train_rmse = compute_rmse(train_loader)
    initial_val_rmse = compute_rmse(val_loader)
    initial_test_rmse = compute_rmse(test_loader)

    # Store the initial RMSE in metrics
    metrics["train_rmse"].append(float(initial_train_rmse))
    metrics["val_rmse"].append(float(initial_val_rmse))
    metrics["test_rmse"].append(float(initial_test_rmse))

    # Print the initial RMSE values
    print(f"Initial Train RMSE: {initial_train_rmse}")
    print(f"Initial Validation RMSE: {initial_val_rmse}")
    print(f"Initial Test RMSE: {initial_test_rmse}")

    # Training loop
    for epoch in range(3):
        model.train()
        train_losses = []
        train_preds = []
        train_labels = []

        for batch in train_loader:
            inputs = {key: val.to(device) for key, val in batch.items() if key != "labels"}
            labels = batch["labels"].to(device).to(torch.float16)

            outputs = model(**inputs)
            # Extract hidden states and pass through regression head
            hidden_states = outputs.hidden_states[-1]  # Shape: [batch_size, sequence_length, hidden_size]
            #pooled_output = hidden_states[:, 0, :]  # Use the first token's hidden state
            pooled_output = hidden_states.mean(dim=1).to(device)  # Shape: [batch_size, hidden_size] 
            logits = model.regression_head(pooled_output)  # Shape: [batch_size, 1]

            loss = loss_fn(logits, labels).to(torch.float16)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_losses.append(loss.item())
            train_preds.extend(logits.squeeze(-1).detach().cpu().numpy())
            train_labels.extend(labels.squeeze(-1).detach().cpu().numpy())

        train_rmse = mean_squared_error(train_labels, train_preds, squared=False)
        metrics["train_rmse"].append(float(train_rmse))
        metrics["train_loss"].append(float(sum(train_losses) / len(train_losses)))

        # Validation
        model.eval()
        val_losses = []
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for batch in val_loader:
                inputs = {key: val.to(device) for key, val in batch.items() if key != "labels"}
                labels = batch["labels"].to(device).to(torch.float16)

                outputs = model(**inputs)


                hidden_states = outputs.hidden_states[-1]  # Shape: [batch_size, sequence_length, hidden_size]
                pooled_output = hidden_states.mean(dim=1).to(device)  # Shape: [batch_size, hidden_size]

                # Pass through the regression head
                logits = model.regression_head(pooled_output)  # Shape: [batch_size, 1]

                val_loss = loss_fn(logits, labels)
                val_losses.append(val_loss.item())
                val_preds.extend(logits.squeeze(-1).detach().cpu().numpy())
                val_labels.extend(labels.squeeze(-1).detach().cpu().numpy())

        val_rmse = mean_squared_error(val_labels, val_preds, squared=False)
        metrics["val_rmse"].append(float(val_rmse))
        metrics["val_loss"].append(float(sum(val_losses) / len(val_losses)))

        print(f"Epoch {epoch + 1}, Train RMSE: {train_rmse}, Val RMSE: {val_rmse}")

        # Test Evaluation
        model.eval()
        test_losses = []
        test_preds = []
        test_labels = []

        with torch.no_grad():
            for batch in test_loader:
                inputs = {key: val.to(device) for key, val in batch.items() if key != "labels"}
                labels = batch["labels"].to(device).to(torch.float16)

                outputs = model(**inputs)

                # Extract hidden states and pass through regression head
                hidden_states = outputs.hidden_states[-1]  # Get the last layer's hidden states

                pooled_output = hidden_states.mean(dim=1).to(device)  # Shape: [batch_size, hidden_size]
                logits = model.regression_head(pooled_output)  # Shape: [batch_size, 1]

                # Compute test loss
                test_loss = loss_fn(logits, labels)
                test_losses.append(test_loss.item())

                # Collect predictions and labels for metrics
                test_preds.extend(logits.squeeze(-1).detach().cpu().numpy())  # Ensure 1D array
                test_labels.extend(labels.squeeze(-1).detach().cpu().numpy())  # Ensure 1D array

        test_rmse = mean_squared_error(test_labels, test_preds, squared=False)
        metrics["test_rmse"].append(float(test_rmse))
        metrics["test_loss"].append(float(sum(test_losses) / len(test_losses)))

        print(f"Test RMSE: {test_rmse}, Test Loss: {metrics['test_loss']}")

    print("Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    # Save metrics to JSON
    test_results_dir = os.path.join(ROOT_PATH, "test_results")
    os.makedirs(test_results_dir, exist_ok=True)
    results_file = os.path.join(test_results_dir, f"trial_{trial.number}_metrics.json")

    # Convert float16 values to float
    #metrics = prepare_metrics_for_json(metrics)

    with open(results_file, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"Metrics saved to {results_file}")

    # Return the average validation loss
    return model, tokenizer, sum(metrics["val_loss"]) / len(metrics["val_loss"])

# # Run the hyperparameter optimiztion
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


# # Save model
# # PRAVEEN: commenting out for now just to test the training code
# # Save the best model
# best_trial = study.best_trial
# print(f"Best trial: {best_trial.number}, Best trials params: {best_trial.params}")

# # Reload tokenizer and prompt generator for saving
# save_path = f"{LORA_REGRESSION_HEAD_PATH}_best_trial"
# os.makedirs(save_path, exist_ok=True)
# best_tokenizer.save_pretrained(save_path)
# best_model.save_pretrained(save_path)
# print(f"Trained regression head saved to {save_path}")

# reloaded_model = AutoModelForCausalLM.from_pretrained(save_path)
# reloaded_tokenizer = AutoTokenizer.from_pretrained(save_path)
# print("Reloaded model and tokenizer successfully.")

# # BECCA: Old save code below
# # save_path = f"{LORA_REGRESSION_HEAD_PATH}_trial_{study.best_trial.number}"

# # os.makedirs(save_path, exist_ok=True)
# # model.save_pretrained(save_path)
# # print(f"Model saved to {save_path}")

def wrapped_objective(trial):
    """
    Wrapper for the Optuna objective function to store the best model and tokenizer globally.
    """
    global best_model, best_tokenizer
    model, tokenizer, val_loss = objective(trial)
    best_model = model
    best_tokenizer = tokenizer
    return val_loss


def main():
    """
    Main function to run hyperparameter optimization and save the best model.
    """
    global best_model, best_tokenizer

    # Run the hyperparameter optimization
    study = optuna.create_study(direction='minimize')
    study.optimize(wrapped_objective, n_trials=10)

    # Save the best model
    best_trial = study.best_trial
    print(f"Best trial: {best_trial.number}, Best trial params: {best_trial.params}")

    # Define save path
    save_path = f"{LORA_REGRESSION_HEAD_PATH}_best_trial"
    os.makedirs(save_path, exist_ok=True)

    # Save tokenizer and model
    best_tokenizer.save_pretrained(save_path)
    best_model.save_pretrained(save_path)

    torch.save(best_model.regression_head.state_dict(), REGRESSION_HEAD_PATH)
    print(f"Trained regression head saved to {save_path}")

    # Reload the model and tokenizer to verify saving
    reloaded_model = AutoModelForCausalLM.from_pretrained(save_path).to(device)  # Move model to GPU
    reloaded_model.config.output_hidden_states = True  # Enable hidden states
    reloaded_tokenizer = AutoTokenizer.from_pretrained(save_path)

    # Re-add the regression head
    reloaded_model.regression_head = torch.nn.Linear(
        reloaded_model.config.hidden_size, 1
    ).to(device).to(torch.float16)

    # Load the saved regression head weights
    reloaded_model.regression_head.load_state_dict(torch.load(REGRESSION_HEAD_PATH))

    print("Reloaded model and tokenizer successfully.")

    # Test inference
    print("\nRunning inference on a sample input...")
    sample_base_prompt = "Create a learning guide for third graders on how to write a story."
    sample_prompt_variation = "Provide a detailed and step-by-step learning guide on how to write a story for third graders. Be sure to include examples and structure."
    combined_input = f"{sample_base_prompt} {sample_prompt_variation}"

    # Tokenize the input
    inputs = reloaded_tokenizer(
        combined_input,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    ).to(device)  # Move inputs to the same device as the model

    # Perform inference
    with torch.no_grad():
        outputs = reloaded_model(**inputs)
        hidden_states = outputs.hidden_states[-1]  # Extract the last hidden states
        pooled_output = hidden_states.mean(dim=1).to(torch.float16).to(device)  # Average pooling
        logits = reloaded_model.regression_head(pooled_output)  # Pass through regression head
        prediction = logits.squeeze(-1).cpu().item()  # Convert to scalar

    print(f"Sample Input: {combined_input}")
    print(f"Predicted Score: {prediction}")


if __name__ == "__main__":
    main()