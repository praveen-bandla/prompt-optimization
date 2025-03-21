
from src.utils.prompt import *
from src.utils.data_handler import *
from configs.root_paths import *
import yaml
import json

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import sys


# Step 1: Collect the instruction for generating base prompts
# Step 2: Run inference to collect all the base prompts
# Step 3: Write the base prompts to a SQLite database


# Step 1: Collect the instruction for generating base prompts
def load_instruction():
    if not os.path.exists(BASE_PROMPT_MODEL_INPUT):
        raise FileNotFoundError(f"Instruction file not found at {BASE_PROMPT_MODEL_INPUT}")
    with open(BASE_PROMPT_MODEL_INPUT, 'r') as file:
        return file.read()

