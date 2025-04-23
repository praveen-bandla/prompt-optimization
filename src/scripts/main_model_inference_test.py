"""
Main Model Inference for Test

This script will contain code to automate the procedure of running main model inference on test base prompts. It will take as input a start and end bp_idx, and will run inference on all prompt variations for each bp_idx that falls between the start and end. The output will be written to a Parquet file in the respective location for each bp_idx, as outlined in the README.

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
- `main_model_input.json`: A JSON file containing instructions for inference.
- `main_model_config.yaml`: A YAML file specifying model parameters and settings.
- `main_model_inference.py`: The script that performs the main model inference.
"""

from src.utils.prompt import *
from src.utils.data_handler import *
from configs.root_paths import *
import yaml
import json

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
#import argparse
import sys
    

def load_configs():
    '''
    Reads the model configuration file.
    '''
    config_path = MAIN_MODEL_CONFIGS

    with open(config_path, 'r') as f:
        configs = yaml.safe_load(f)
      
    return configs

def load_model():
    '''
    Loads the main model and tokenizer.
    '''

    model = AutoModelForCausalLM.from_pretrained(
        MAIN_MODEL_ID,
        torch_dtype="auto",
        trust_remote_code=True, # have to use for huggingface models
        device_map = "auto"
        )
    
    tokenizer = AutoTokenizer.from_pretrained(MAIN_MODEL_ID)

    pipe = pipeline(
        'text-generation', 
        model=model, 
        tokenizer=tokenizer
        )

    return pipe


def construct_model_input(bp_obj):
    '''
    Reads the model input text file and place the template.
    '''

    if not os.path.exists(MAIN_MODEL_INPUT):
        raise FileNotFoundError(f'Main Model input instruction json file not found at {MAIN_MODEL_INPUT}')
    with open(MAIN_MODEL_INPUT, 'r') as f:
        prompt_structure = json.load(f)

    system_role = prompt_structure["system_role"]
    content_template = prompt_structure["content_template"]

    print(type(pv_obj))

    bp_str = bp_obj.get_prompt_str()

    print(f"Prompt variation string: {bp_str}")
    print(f'Pv str type: {type(bv_str)}')
    content = content_template.format(bp_str_template = bp_str)

    full_prompt = [
        {
            "role": "system",
            "content": system_role
        },
        {
            "role": "user",
            "content": content
        }
    ]

    return full_prompt


def collect_base_prompts(bp_idx_start, bp_idx_end):
    '''
    Collects the base prompt strings from the start and end indices provided.
    '''
    bp_data_handler = BasePromptDB()

    bps = []

    for i in range(bp_idx_start, bp_idx_end):
        prompt = bp_data_handler.fetch_prompt(i)
        bp_obj = BasePrompt(i)
        bps.append(bp_obj)

    return bps

def main_model_inference_for_bps(bp_idx_start, bp_idx_end, pipe, configs):
    '''
    Performs main model inference on all base prompts for the given indexes in batches.
    '''
    all_bp_outputs = []
    bps = collect_base_prompts(bp_idx_start, bp_idx_end)
    batch_size = MM_TEST_BATCH_SIZE

    max_new_tokens = configs.get("max_new_tokens")
    temperature = configs.get("temperature")
    do_sample = configs.get("do_sample")
    generation_args = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "do_sample": do_sample,
        "return_full_text": False
    }

    for i in range(bp_idx_start, bp_idx_end, batch_size):
        mo_parquet = ModelOutputParquet(i, MODEL_TEST_OUTPUTS)

        batch_bps = bps[i:i+batch_size]
        prompts = [construct_model_input(bp) for bp in batch_bps]
        outputs = pipe(prompts, **generation_args)
        
        print(f"DEBUG: outputs sample: {outputs[:2]}")

        for bp, output in zip(batch_bps, outputs):
            # Fix: Handle nested list case
            if isinstance(output, list):
                output = output[0]
            model_output = output['generated_text']
            all_bp_outputs.append((bp.bp_idx, model_output))

        print(f"Processed batch {i // batch_size + 1} of bp_idx {bp_idx}")

        mo_parquet.insert_model_outputs(all_bp_outputs)
        print(f"Completed bp_idx {bp_idx} with {len(all_bp_outputs)} outputs.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise ValueError("Please provide a list of base prompt indices.")

    start_idx = int(sys.argv[1])
    end_idx = int(sys.argv[2])

    pipe = load_model()
    configs = load_configs()

    for bp_idx in range(start_idx, end_idx):
        main_model_inference_per_base_prompt(bp_idx, pipe, configs)
        print(f"Finished inference for base prompt index {bp_idx}.")

    print("All inferences have been completed.")