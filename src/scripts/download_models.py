from src.utils.prompt import *
from src.utils.data_handler import *
from configs.root_paths import *

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import os

# Define the local directory to store the model

# change this as needed
# net_id = "pb3060"
# local_model_dir = f"/scratch/{net_id}/prompt-optimization/models/microsoft-Phi-3.5-mini-instruct/"

# # Download the model and tokenizer
# model = AutoModelForCausalLM.from_pretrained(
#     "microsoft/Phi-3.5-mini-instruct", 
#     cache_dir=local_model_dir,
#     trust_remote_code=True
# )
# tokenizer = AutoTokenizer.from_pretrained(
#     "microsoft/Phi-3.5-mini-instruct", 
#     cache_dir=local_model_dir,
#     trust_remote_code=True
# )


# print("Model and tokenizer downloaded and saved locally.")

### Prompt Optimizer Section ###

# Shared tokenizer
# regression_head_tokenizer = AutoTokenizer.from_pretrained(REGRESSION_HEAD_BASE_MODEL_ID)
# prompt_gen_tokenizer = AutoTokenizer.from_pretrained(PROMPT_GEN_BASE_MODEL_ID)

# # Load regression head model base
# regression_head_base = AutoModelForCausalLM.from_pretrained(
#     REGRESSION_HEAD_BASE_MODEL_ID,
#     torch_dtype = torch.float16,
#     device_map="auto",
#     cache_dir = REGRESSION_HEAD_BASE_PATH
# )
# regression_head_base.save_pretrained(LORA_REGRESSION_HEAD_PATH)
# print("Base model for regression head saved to:", LORA_REGRESSION_HEAD_PATH)

# Load prompt generator model base
prompt_gen_base = AutoModelForCausalLM.from_pretrained(
    PROMPT_GEN_BASE_MODEL_ID,
    torch_dtype = torch.float16,
    device_map="auto",
    cache_dir = PROMPT_GEN_BASE_PATH
)
prompt_gen_base.save_pretrained(LORA_PROMPT_GEN_PATH)
print("Base model frompt generator saved to:", LORA_PROMPT_GEN_PATH)

# NOTE: Using device_map="auto" will automatically place the model on the same GPU if memory allows. Must ensure Greene job requests enough memory (24GB+)

