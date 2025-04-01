'''
Generate Prompt Varitions (New)

This updated script generates a set of prompt variations by calling the prompt variation model to produce prompt variations based on the stored base prompts. It runs inferences and generates 'NUM_PROMPT_VARIATIONS' prompt variations for further use.

Inputs:
    bp_idx: The index of the base prompt to generate prompt variations for.

Dependencies:
    - prompt_variation_model_input.json: A JSON file containing instructions for generating base prompts.
    - prompt_variation_model_config.yaml: A YAML file specifying model parameters and settings.
    - data_handler.py: prompt variation class
    - prompt.py: prompt variation class

Outputs:
    None

Process:
1. Collect the instruction for generating prompt variations. Uses the NUM_PROMPT_VARIATIONS from the data_size_configs file and the prompt_variation_model_input.json file for the instruction.
2. Loads the model
3. Loads the model configuration file
4. Runs inference to collect all the prompt variations
5. Writes the prompt variations to a single parquet associated to the given base prompt
'''

from src.utils.prompt import *
from src.utils.data_handler import *
from configs.root_paths import *
import yaml
import json
import random
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Step 1: Collect the instruction for generating prompt variations
def collect_instruction():
    '''
    Creates instructions for generating prompt variations. Uses the prompt_variation_model_input.json file as the model input, along with NUM_PROMPT_VARIATIONS from the data_size_configs file.

    Inputs: None

    Outputs:
        - full_prompt: A list containing the system role and content template for generating prompt variations, stored as two dictionaries.
    
    '''
    if not os.path.exists(PROMPT_VARIATION_MODEL_INPUT):
        raise FileNotFoundError(f"Instruction file not found at {PROMPT_VARIATION_MODEL_INPUT}")
    with open(PROMPT_VARIATION_MODEL_INPUT, 'r') as f:
        prompt_structure = json.load(f)
    
    system_role = prompt_structure["system_role"]
    content_template = prompt_structure["content_template"]

    # Replace placeholders with actual values
    content_template = content_template.replace("{bp_str}", #BP_STR)
    content_template = content_template.replace("{num_prompt_variations}", str(NUM_PROMPT_VARIATIONS / 2))
    #instruction = system_role + " " + content_template

    content = content_template.format(num_prompt = NUM_PROMPT_VARIATIONS)

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
    # bp_data_handler = BasePromptDB(SQL_DB)
    # bp = BasePrompt(bp_idx, bp_data_handler, instruction)
    # return bp

def load_configs():
    '''
    Reads the model configuration file.

    Inputs: None

    Outputs: 
        - configs: the configurations for the prompt variation model.
    '''
    config_path = PROMPT_VARIATION_MODEL_CONFIG

    with open(config_path, 'r') as f:
        configs = yaml.safe_load(f)
      
    return configs

def load_model():
    '''
    Loads the prompt variation model for inference. 
    '''
    model = AutoModelForCausalLM.from_pretrained(
        PROMPT_VARIATION_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True
    )

    tokenizer = AutoTokenizer.from_pretrained(PROMPT_VARIATION_MODEL, trust_remote_code=True)

    pipe = pipeline(
        'text-generation', 
        model=model, 
        tokenizer=tokenizer, 
        device_map="auto"
        )

    return pipe

# Step 2: Run inference to collect all the prompt variations
def prompt_variation_inference():
    '''
    Runs inference on the prompt_variation_model to generate the desired output. It solely retrievers the response as a string and does not process it further.
    '''
    instruction = collect_instruction()
    pipe = load_model()
    configs = load_configs()

    max_tokens = configs.get("max_tokens")
    temperature = configs.get("temperature")
    top_p = configs.get("top_p")
    do_sample = configs.get("do_sample") # check this exists

    generation_args = {
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "do_sample": do_sample
    }

    print("Successfully loaded model and configs.")
    print("Running inference to generate prompt variations...")
    #outputs = pipe(instruction, **generation_args)
    outputs = pipe(instruction)
    print("Inference complete.")
    print("Outputs: ", outputs[0]["generated_text"])
    return outputs[0]["generated_text"]

def parse_model_output_as_pv_objects(model_output):
    '''
    This function parses the model output to extract the prompt variations as tuples of (bpv_idx, prompt_variation_string).

    Inputs:
        - model_output: The model output string from prompt_variation_inference()

    Outputs:
        - list: A list of tuples, each containing the base prompt variation index and the prompt variation string. Stored as a list of (bpv_idx, bpv_str)
    '''
    base_prompts = json.loads(model_output)

    # # creating random order of prompts stored as int indices
    # random_indices = random.sample(range(NUM_BASE_PROMPTS), NUM_BASE_PROMPTS)

    # returning a list of tuples in the desired format of the random order of prompts
    return [(new_idx, base_prompts[random_idx]) for new_idx, random_idx in enumerate(random_indices)]