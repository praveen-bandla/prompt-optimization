"""
Generate Base Prompts

This script generates a set of base prompts by calling a model to produce base prompts based on
provided input. It runs inference and generates 'n' base prompts for further use.

No input: None, base prompts generated as a batch

Load as dependencies:
- `base_prompt_input.txt`: A text file containing the base instruction for generating prompts.
- `base_prompt_config.yaml`: A YAML file with configuration settings for prompt generation.
- data_handler.py base prompt class
- prompt.py base prompt class

Outputs: None

Overview:
Script pulls number of base prompts from configs, generates prompts, randomly shuffles them, and writes them to a SQLite database. 
  | bp_idx | base_prompt |
  |--------|-------------|
  | 1 | "Generated base prompt 1" |
  | 2 | "Generated base prompt 2" |
  | ...    | ...         |

Shuffling implementation:
- DON'T run through list of strings and randomly shuffle and write to SQL
- Shuffle indices 0 to n-1, then write:
for i in [shuffled_indices]:
    write to SQL strs[i]

- Opens and writes the generated prompts to `base_prompts.sqlite` in SQLite/database format for efficient storage. 

Usage:
- Ensure that both input files (`base_prompt_input.txt` and `base_prompt_config.yaml`) are in the same
  directory as the script before running it.
- The generated prompts will be saved to a SQLite file named `base_prompts.sqlite`.
- Have slurm config file ready to run the script on HPC
- Add all libraries I'm importing into requirements.txt

Dependencies:
- [List of libraries or modules, e.g. 'pandas', 'transformers', 'torch', etc.]
"""

import sys
import os
import yaml
import random
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Add project root (prompt-optimization/) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

# Load configuration
def load_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

# Load instruction text for LLM input
def load_instruction(input_path):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Instruction file not found at {input_path}")
    with open(input_path, 'r') as file:
        return file.read()

def call_llm_generate_prompts(prompt, config):
    model_path = "/Users/darrenjian/.llama/checkpoints/Llama3.1-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    output = model.generate(
        inputs["input_ids"],
        max_length=config.get("max_length", 2048),
        num_return_sequences=1,
        temperature=config.get("temperature", 0.7),
        top_p=config.get("top_p", 0.9),
        do_sample=True
    )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    return decoded

def parse_prompts(generated_text):
    prompts = [line.strip() for line in generated_text.split('\n') if line.strip()]
    return prompts

def main():
    # Define paths
    input_path = os.path.join(MODEL_INPUT_PATH, "base_prompt_input.txt")
    config_path = os.path.join(CONFIGS_PATH, "model_configs/base_prompt_config.yaml")

    # Load input + config
    instruction_text = load_instruction(input_path)
    config = load_config(config_path)

    if config is None:
        raise ValueError("Configuration could not be loaded. Check your YAML file.")

    n = config.get("num_base_prompts", 1000)

    # Call model to generate prompts
    generated_output = call_llm_generate_prompts(instruction_text, config)
    base_prompts = parse_prompts(generated_output)

    if len(base_prompts) < n:
        raise ValueError(f"Model returned fewer prompts ({len(base_prompts)}) than requested ({n}).")

    # Shuffle indices
    indices = list(range(n))
    random.shuffle(indices)

    # Write to SQLite database
    db = BasePromptDB()
    for idx in indices:
        bpv_idx = (idx, -1)  # base prompts use variation index -1
        prompt = base_prompts[idx]
        bp = BasePrompt(bpv_idx)
        bp.prompt = prompt
        bp.save_base_prompt(db)
    db.close_connection()

    print(f"Successfully saved {n} base prompts to base_prompts.sqlite")

if __name__ == "__main__":
    main()