"""
Main Model Inference

This script will contain code to automate the procedure of running main model inference on a batch of given prompts. It will take as input a start and end bp_idx, and will run inference on all prompt variations for each bp_idx that falls between the start and end. The output will be written to a Parquet file in the respective location for each bp_idx, as outlined in the README.

Inputs:
    - start_idx (int): The starting base prompt index. This is the first base prompt index for which inference will be run.
    - end_idx (int): The ending base prompt index. The script runs on all base prompt indices up to end_idx, excluding end_idx itself.

Example usage:
    # Run inference on all prompt variations for base prompt indices 0,1,2,3,4.
    python main_model_inference.py 0 5


Outputs:
- Writes the generated output to a Parquet file named `{i]_model_outputs.parquet`, which contains results  for prompts indexed from `(i, -1)` to `(i, n)`, where `n` is the number of partitions. `i` is the index of the base prompt.
  Example format of a single file:
  | bpv_idx | main model output |
  |---------|------------------|
  | (0, -1)  | "Model output 1" |
  | (0, 0)  | "Model output 2" |
  | ...     | ...              |

  This file would contain all model outputs for all prompt variations for bp_idx (base prompt index) 0.

Process:
1. Reads the bp_idx as a script parameter
2. Collects the prompt variations for the given bp_idx
3. Collects the model input text file and configuration file
4. Performs inference using the provided prompt variations
5. Opens or creates the corresponding Parquet file

Dependencies:
- `main_model_input.txt`: A text file containing instructions for inference.
- `main_model_config.yaml`: A YAML file specifying model parameters and settings.
- `main_model_inference.py`: The script that performs the main model inference.
"""

from src.utils import prompt
from src.utils import data_handler
from configs.root_paths import *
import yaml

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
#import argparse
import sys

# Step 2: Collect the prompt variations for the given bp_idx
def collect_prompt_variations(bp_idx):
    '''
    Collects the prompt variation strings for the bp_idx.
    '''
    pv_data_handler = data_handler.PromptVariationParquet(bp_idx)
    bpv_idxs, pv_strs = pv_data_handler.fetch_all_variations()
    num_pvs = len(bpv_idxs)

    pvs = []

    for i in num_pvs:
        pv_obj = prompt.PromptVariation(bpv_idxs[i], pv_strs[i])
        pvs.append(pv_obj)

    return pvs

def load_model():
    '''
    Loads the main model and tokenizer.
    '''

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("CUDA (GPU) is not available. Please check your setup.")
    tokenizer = AutoTokenizer.from_pretrained(MAIN_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MAIN_MODEL,
        torch_dtype=torch.float16,
        device_map={"": 0} # Apparently huggingface wants us to use device_map
    )
    return model, tokenizer


def construct_model_input(pv_obj):
    '''
    Reads the model input text file and place the template.
    '''
    with open(MAIN_MODEL_INPUT, 'r') as f:
        model_input = f.read()

    pv_str = pv_obj.fetch_variation_str()
    # note to self: pv_str_template is a template string, so we need to replace the placeholder with the actual prompt variation string
    prompt = model_input.format(pv_str_template = pv_str)

    return prompt
    

def model_config():
    '''
    Reads the model configuration file.
    '''
    config_path = MAIN_MODEL_CONFIGS

    with open(config_path, 'r') as f:
        configs = yaml.safe_load(f)
      
    return configs



def main_model_inference_per_prompt_variation(pv_obj):
    '''
    Performs inference using the provided prompt variation.

    Input:
    - pv_obj (PromptVariation): The prompt variation object.

    Returns:
    - str: The model output string.
    '''
    prompt = construct_model_input(pv_obj)
    model, tokenizer = load_model()

    configs = model_config()

    # reading all relevant configs
    max_length = configs.get("max_length")
    temperature = configs.get("temperature")
    top_p = configs.get("top_p")
    top_k = configs.get("top_k")
    repetitive_penalty = configs.get("repetitive_penalty")

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetitive_penalty=repetitive_penalty
        )

    return tokenizer.decode(output[0], skip_special_tokens=True)


def main_model_inference_per_base_prompt(bp_idx):
    '''
    Performs main model inference on all prompt variations for the given bp_idx. Stores all the outputs in its respective Parquet file.

    Input:
    - bp_idx (int): The base prompt index.
    '''
    all_pv_outputs = []
    mo_parquet = prompt.ModelOutputParquet(bp_idx)
    for pv_obj in collect_prompt_variations(bp_idx):
        model_output = main_model_inference_per_prompt_variation(pv_obj)
        all_pv_outputs.append((pv_obj.get_prompt_index(), model_output))
    mo_parquet.insert_model_outputs(all_pv_outputs)


if __name__ == "main":
    if len(sys.argv) != 3:
        raise ValueError("Please provide a list of base prompt indices.")
        sys.exit(1)

    start_idx = int(sys.argv[1])
    end_idx = int(sys.argv[2])

    for bp_idx in range(start_idx, end_idx):
        main_model_inference_per_base_prompt(bp_idx)
        print(f"Finished inference for base prompt index {bp_idx}.")
    print("All inferences have been completed.")



    



    

