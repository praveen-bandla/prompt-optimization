"""
Generate Prompt Variations

This script generates a set of prompt variations by calling the prompt variation model to run inference/generate 'n' prompt variations. 

Inputs:
- `prompt_variation_input.txt`: A text file containing the base instruction for generating prompt variations.
- `prompt_variation_config.yaml`: A YAML file with configuration settings for prompt variation generation.
- Section of prompt SQL DB from `generate_base_prompts.py`, bp_idx from 0 to n, n defined in config file

Outputs:
- Opens and writes the generated prompts to a series of Parquet files, one per base prompt.
- The output files will be named `prompt_variations_{n}.parquet`, where {n} is the number of the base prompt.
  The output includes `bpv_idx`, which is a tuple in the format `(bp_idx, index)`, where `index` is an integer 
  between 1 and `n`, representing the nth variation. Each entry will also include a string `prompt_variation: str`.
  Example format:
  | bpv_idx | prompt_variation |
  |---------|------------------|
  | (1, 1) | "Generated prompt variation 1" |
  | (1, 2) | "Generated prompt variation 2" |
  | ...     | ...              |

Overview:
- Inference prompt_variation_config.yaml and prompt_variation_input.txt to generate variations for a given base prompt and write to a .parquet file using functions from data_handler.py
- function to generate prompt variations takes one bp_idx number
  - Creates one PV-Parquet object (handles info from one specific prompt variation)
  - Creates 200 PV objects using strings from inferencing
- this only needs to do for one base prompt, we will call this script in a for loop for all base prompts we're looking at

Usage:
- Ensure that both input files (`base_prompt_input.txt` and `base_prompt_config.yaml`) are in the same
  directory as the script before running it.
- The generated prompts will be saved to a Parquet file named `prompt_variations_{n}.parquet`.

Dependencies:
- [List of libraries or modules, e.g. 'pandas', 'transformers', 'torch', etc.]
- Needs the index of the base prompt
"""
import sys
import os
import yaml
import random
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from utils.data_handler import BasePromptDB, PromptVariationParquet
from utils.prompt import PromptVariation
from configs.root_paths import *

# Load configuration
def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def load_instruction(path):
    with open(path, 'r') as f:
        return f.read()

def call_llm_variation_generator(full_prompt, config):
    model_path = config.get("model_path", "DEEPSEEK-R1") # edit this to the model path
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")

    inputs = tokenizer(full_prompt, return_tensors="pt").to("cuda")

    outputs = model.generate(
        inputs["input_ids"],
        max_length=config.get("max_length", 2048),
        num_return_sequences=1,
        temperature=config.get("temperature", 0.7),
        top_p=config.get("top_p", 0.9),
        do_sample=True
    )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return [line.strip() for line in decoded.split('\n') if line.strip()]

def main():
    # Ensure the user provides a base prompt index as a command-line argument
    if len(sys.argv) < 2:
        print("Usage: python generate_prompt_variations.py <bp_idx>")
        sys.exit(1)

    # Parse and prepare the base prompt index
    bp_idx = int(sys.argv[1])
    bpv_base = (bp_idx, -1)  # The base prompt uses a -1 variation index

    # Load the base prompt string from the SQLite database
    db = BasePromptDB()
    base_prompt = db.fetch_prompt(bpv_base)
    db.close_connection()

    if base_prompt is None:
        raise ValueError(f"No base prompt found for index {bp_idx}")

    # Load the model input instruction and YAML config
    instruction_path = os.path.join(MODEL_INPUT_PATH, "prompt_variation_input.txt")
    config_path = os.path.join(CONFIGS_PATH, "model_configs/prompt_variation_config.yaml")

    instruction = load_instruction(instruction_path)
    config = load_config(config_path)

    # Combine instruction and base prompt into a single LLM input
    full_prompt = instruction + "\n" + base_prompt

    # Call the LLM to generate candidate variations
    generated_variations = call_llm_variation_generator(full_prompt, config)

    # Validate the count of generated variations
    n = config.get("num_prompt_variations", 200)
    if len(generated_variations) < n:
        raise ValueError(f"Model returned fewer variations ({len(generated_variations)}) than requested ({n}).")

    # Truncate to exactly n if more are returned
    generated_variations = generated_variations[:n]

    # Use PromptVariationParquet to write variations to disk
    pv_handler = PromptVariationParquet()
    pv_data = []

    for i, variation in enumerate(generated_variations):
        bpv_idx = (bp_idx, i)
        pv = PromptVariation(bpv_idx, pv_handler, variation)
        pv_data.append((pv.bpv_idx, pv.full_string))

    pv_handler.insert_prompt_variations(pv_data)
    print(f"Successfully saved {n} variations for base prompt {bp_idx}.")

if __name__ == "__main__":
    main()
