"""
Main Model Inference

This script will contain code to automate the procedure of running main model inference on a batch of given prompts. It will take as input a list of bpv_idx, generate main model output accordingly, and store into corresponding Parquet file. 

Inputs:
- List of bpv_idx: A list of integers representing multiple prompt variations' indices.

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

# Step 2: Collect the prompt variations for the given bp_idx
def collect_prompt_variations(bp_idx):
    '''
    Collects the prompt variation strings for the bp_idx.
    '''
    pv_data_handler = data_handler.PromptVariationDataHandler()
    bpv_idxs, pv_strs = pv_data_handler.fetch_all_variations(bp_idx)
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


def model_input_text():
    '''
    Reads the model input text file.
    '''

def construct_prompt():
    '''
    Generate
    '''
    pass
    

def model_config():
    '''
    Reads the model configuration file.
    '''
    config_path = MAIN_MODEL_CONFIGS

    with open(config_path, 'r') as f:
        configs = yaml.safe_load(f)
      
    return configs



def main_model_inference_per_prompt_variation(prompt):
    '''
    Performs inference using the provided prompt variation.
    '''
    model, tokenizer = load_model()

    configs = model_config()

    # reading all relevant configs
    max_length = configs.get("max_length")
    temperature = configs.get("temperature")
    top_p = configs.get("top_p")

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p
        )

    return tokenizer.decode(output[0], skip_special_tokens=True)


def main_model_inference_per_base_prompt(bp_idx):
    '''
    Performs main model inference on all prompt variations for the given bp_idx.
    '''
    pass


if __name__ == "main":
    # when this script is called, it is expected to be called with a list of bp_idxs

    pass


    

