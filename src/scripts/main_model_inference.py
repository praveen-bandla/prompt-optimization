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

# Step 2: Collect the prompt variations for the given bp_idx
def collect_prompt_variations(bp_idx):
    '''
    Collects the prompt variation strings for the bp_idx.
    '''
    pv_data_handler = PromptVariationParquet(bp_idx)
    bpv_idxs, pv_strs = pv_data_handler.fetch_all_variations()
    num_pvs = len(bpv_idxs)

    pvs = []

    for i in range(num_pvs):
        pv_obj = PromptVariation(bpv_idx = bpv_idxs[i], pv_parquet = pv_data_handler, variation_str = pv_strs[i])
        bpv_idx = pv_obj.get_bpv_idx()
        pvs.append(pv_obj)

    return pvs

def load_model():
    '''
    Loads the main model and tokenizer.
    '''

    # # old generic implementation. Reworking below for PHI-3.5 specifically, based on the huggingface docs
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # if device != "cuda":
    #     raise RuntimeError("CUDA (GPU) is not available. Please check your setup.")
    # tokenizer = AutoTokenizer.from_pretrained(MAIN_MODEL)
    # model = AutoModelForCausalLM.from_pretrained(
    #     MAIN_MODEL,
    #     torch_dtype=torch.float16,
    #     device_map={"": 0} # Apparently huggingface wants us to use device_map
    # )
    # return model, tokenizer

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


def construct_model_input(pv_obj):
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

    pv_str = pv_obj.get_prompt_variation_str()

    print(f"Prompt variation string: {pv_str}")
    print(f'Pv str type: {type(pv_str)}')
    content = content_template.format(pv_str_template = pv_str)

    full_prompt = [
        {
            "role": "system",
            #"content": [{"type": "text", "text": system_role}]
            "content": system_role # apparently implementation for Phi-3.5 is different to OpenAI 4o
        },
        {
            "role": "user",
            #"content": [{"type": "text", "text": content}]
            "content": content
        }
    ]

    return full_prompt
    

def load_configs():
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

    #return "Sample model output"
    prompt = construct_model_input(pv_obj)
    #model, tokenizer = load_model()
    pipe = load_model()

    configs = load_configs()

    # new configs that are relevant to PHI-3.5
    max_new_tokens = configs.get("max_new_tokens")
    temperature = configs.get("temperature")
    do_sample = configs.get("do_sample") # decides whether to use sampling or greedy decoding (diverse outputs vs. best outputs)

    # old implementation
    # inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    # with torch.no_grad():
    #     output = model.generate(
    #         **inputs,
    #         max_length=max_length,
    #         temperature=temperature,
    #         top_p=top_p,
    #         top_k=top_k,
    #         repetitive_penalty=repetitive_penalty
    #     )
    # return tokenizer.decode(output[0], skip_special_tokens=True)

    # new implementation
    generation_args = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "do_sample": do_sample,
        "return_full_text": False # determines whether to the prompt as part of the output.

    }

    output = pipe(prompt, **generation_args)
    return output[0]['generated_text']


def main_model_inference_per_base_prompt(bp_idx, pipe, configs):
    '''
    Performs main model inference on all prompt variations for the given bp_idx in batches.
    '''
    all_pv_outputs = []
    mo_parquet = ModelOutputParquet(bp_idx)
    prompt_variations = collect_prompt_variations(bp_idx)

    batch_size = MM_OUTPUT_BATCH_SIZE  # Use your defined variable
    max_new_tokens = configs.get("max_new_tokens")
    temperature = configs.get("temperature")
    do_sample = configs.get("do_sample")
    generation_args = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "do_sample": do_sample,
        "return_full_text": False
    }

    for i in range(0, len(prompt_variations), batch_size):
        batch_pvs = prompt_variations[i:i + batch_size]
        prompts = [construct_model_input(pv) for pv in batch_pvs]
        outputs = pipe(prompts, **generation_args)

        for pv, output in zip(batch_pvs, outputs):
            model_output = output['generated_text']
            all_pv_outputs.append((pv.get_bpv_idx(), model_output))

        print(f"🔹 Processed batch {i // batch_size + 1} of bp_idx {bp_idx}")

    mo_parquet.insert_model_outputs(all_pv_outputs)
    print(f"✅ Completed bp_idx {bp_idx} with {len(all_pv_outputs)} outputs.")


    for pv_obj in collect_prompt_variations(bp_idx):
        # model_output = main_model_inference_per_prompt_variation(pv_obj)
        prompt = construct_model_input(pv_obj)
        model_output = pipe(prompt, **generation_args)[0]['generated_text']
        full_bpv_idx = (mo_parquet.get_bp_idx(), pv_obj.get_bpv_idx())
        all_pv_outputs.append((pv_obj.get_bpv_idx(), model_output))
    mo_parquet.insert_model_outputs(all_pv_outputs)


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




    

