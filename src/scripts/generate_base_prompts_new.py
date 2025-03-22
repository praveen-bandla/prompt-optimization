
from src.utils.prompt import *
from src.utils.data_handler import *
from configs.root_paths import *
import yaml
import json

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import sys


# Step 1: Collect the instruction for generating base prompts
# Step 2: Run inference to collect all the base prompts
# Step 3: Write the base prompts to a SQLite database


# Step 1: Collect the instruction for generating base prompts
def collect_instruction(bp_idx):
    if not os.path.exists(BASE_PROMPT_MODEL_INPUT):
        raise FileNotFoundError(f"Instruction file not found at {BASE_PROMPT_MODEL_INPUT}")
    with open(BASE_PROMPT_MODEL_INPUT, 'r') as file:
        instruction = json.load(f)
    
    system_role = instruction["system_role"]
    content_template = instruction["content_template"]
    instruction = system_role + " " + content_template

    bp_data_handler = BasePromptDB(SQL_DB)
    bp = BasePrompt(bp_idx, bp_data_handler, instruction)
    return bp

def load_configs():
    '''
    Reads the model configuration file.
    '''
    config_path = BASE_PROMPT_MODEL_CONFIG

    with open(config_path, 'r') as f:
        configs = yaml.safe_load(f)
      
    return configs

def load_model():
    '''
    Loads the base prompt model for inference. 
    '''
    model = AutoModelForCausalLM.from_pretrained(
        BASE_PROMPT_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_PROMPT_MODEL, trust_remote_code=True)

    pipe = pipeline('text-generation', model=model, tokenizer=tokenizer, device_map="auto")
    return pipe

# Step 2: Run inference to collect all the base prompts

def base_prompt_inference(): # current understanding: this will generate the x (i.e. 100) base prompts needed
    ...

# Step 3: Write the base prompts to a SQLite database
def base_prompt_inference_to_db():
    '''
    Runs the base prompt inference and writes the base prompts to a SQLite database.
    '''
    all_bps = base_prompt_inference()
    db = BasePromptDB(SQL_DB)
    db.write_base_prompts(all_bps)
    db.close()

if __name__ == "Main":
    base_prompt_inference_to_db()
    print("Base prompts generated and written to SQLite database.")
    