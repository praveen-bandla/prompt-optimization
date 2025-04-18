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
prompt_opt_tokenizer = AutoTokenizer.from_pretrained(PROMPT_OPT_BASE_MODEL_ID)

# Load regression head model
regression_head_base = AutoModelForCausalLM.from_pretrained(
    PROMPT_OPT_BASE_MODEL_ID,
    torch_dtype = torch.float16,
    device_map="auto"
)
regression_head_model = PeftModel.from_pretrained(
    regression_head_base,
    LORA_REGRESSION_HEAD_PATH,
    torch_dtype=torch.float16,
    device_map="auto"
)
print("Regression head base and model downloaded and saved locally.")

# Load prompt generator model
prompt_gen_base = AutoModelForCausalLM.from_pretrained(
    PROMPT_OPT_BASE_MODEL_ID,
    torch_dtype = torch.float16,
    device_map="auto"
)
prompt_gen_model = PeftModel.from_pretrained(
    prompt_gen_base,
    LORA_PROMPT_GEN_PATH,
    torch_dtype=torch.float16,
    device_map="auto"
)
print("Regression head base and model downloaded and saved locally.")
# NOTE: Using device_map="auto" will automatically place the model on the same GPU if memory allows. Must ensure Greene job requests enough memory (24GB+)

