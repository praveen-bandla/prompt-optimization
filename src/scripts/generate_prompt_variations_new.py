'''
Generate Prompt Varitions (New)

This updated script generates a set of prompt variations by calling the prompt variation model to produce prompt variations based on the stored base prompts. It runs inferences and generates 'NUM_PROMPT_VARIATIONS' prompt variations for further use.

Inputs:
    bp_idx: A list of integers representing the indices of the base prompts to generate prompt variations for.

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
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import argparse
import re

from huggingface_hub import login

login(token='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')

# PRAVEEN:
'''
to my dearest barren:
open terminal in prompt-optimization directory. run using: PYTHONPATH=$(pwd) python src/scripts/generate_prompt_variations_new.py "[0,1,2]" to avoid import errors
everything barring model inference should work as needed. had to make some changes to data_handler.py's PromptVariationParquet class.

[DONE] 0. (edit) documentation to make it based on a list instead
[WORKS] 1. (edit) Collect instruction -> make it based on bp_idx
[LOOKS GOOD] 2. (review) load_model
[LOOKS GOOD] 3. (review) prompt_variation_inference
[WORKS] 4. (edit) parse_model_output_as_pv_objects -> parse correctly the model output
[WORKS] 5. (create) write_parquet -> write the prompt variations to a parquet file
[WORKS] 6. (create) main () -> call everything based on the input parse
[WORKS] 7. Create if_name__ == "__main__" to run the script with list of bp_idx
'''

# Step 1: Collect the instruction for generating prompt variations
# Define manual formatting function
def collect_instruction(bp_idx):
    '''
    Creates instructions for generating prompt variations based on a base prompt index. Uses the prompt_variation_model_input.json file as the model input, along with NUM_PROMPT_VARIATIONS from the data_size_configs file.

    Inputs:
        - bp_idx: A given integer representing the index of the base prompt to generate prompt variations for.

    Outputs:
        - full_prompt: A list containing the system role and content template for generating prompt variations, stored as two dictionaries.
    
    '''

    # collecting base_prompt_str from the database based on the given index
    bp_db = BasePromptDB()
    bp = BasePrompt(bp_idx)
    bp_str = bp.get_prompt_str()
    

    if not os.path.exists(PROMPT_VARIATION_MODEL_INPUT):
        raise FileNotFoundError(f"Instruction file not found at {PROMPT_VARIATION_MODEL_INPUT}")
    with open(PROMPT_VARIATION_MODEL_INPUT, 'r') as f:
        prompt_structure = json.load(f)
    
    system_role = prompt_structure["system_role"]
    content_template = prompt_structure["content_template"]

    # PRAVEEN EDITS: replacing this replace method with format (not sure if replace works but know for a fact format works.)
    # Replace placeholders with actual values
    # content_template = content_template.replace("{bp_str}", #BP_STR)
    # content_template = content_template.replace("{num_prompt_variations}", str(NUM_PROMPT_VARIATIONS / 2))
    #instruction = system_role + " " + content_template

    content = content_template.format(num_prompt_variations=str(NUM_PROMPT_VARIATIONS // 2), bp_str=bp_str)

    full_prompt = [
        # {
        #     "role": "system",
        #     "content": system_role
        # },
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
        PROMPT_VARIATION_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        # cache_dir = RR_HUGGINGFACE_CACHE
    )

    tokenizer = AutoTokenizer.from_pretrained(
        PROMPT_VARIATION_MODEL_ID, 
        trust_remote_code=True)

    # BECCA: Deepseek needs specifically structured chats so pipeline is not the best choice here
    # pipe = pipeline(
    #     'text-generation', 
    #     model=model, 
    #     tokenizer=tokenizer, 
    #     device_map="auto",
    #     # cache_dir = RR_HUGGINGFACE_CACHE
    #     )

    return model, tokenizer

# Step 2: Run inference to collect all the prompt variations
def prompt_variation_inference():
    '''
    Runs inference on the prompt_variation_model to generate the desired output. It solely retrievers the response as a string and does not process it further.
    '''
    instruction = collect_instruction(bp_idx)
    model, tokenizer = load_model()
    configs = load_configs()

    max_new_tokens = configs.get("max_new_tokens")
    temperature = configs.get("temperature")
    top_p = configs.get("top_p")
    do_sample = configs.get("do_sample")
    repetition_penalty = configs.get("repetition_penalty")

    # Ensure tokenizer has a pad_token and set pad_token_id
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'}) # Add a pad token if it doesn't exist
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token) # Set pad_token_id correctly
    
    # early_stopping = configs.get("early_stopping")
    # eos_token_id = tokenizer.eos_token_id
    # eos_token_id = configs.get("eos_token_id")

    # # Section to ensure EOS and PAD tokens are set correctly - not working as of 040325 5:40
    # if tokenizer.eos_token is None:
    #     tokenizer.eos_token = "</s>"
    # if tokenizer.pad_token is None:
    #     tokenizer.add_special_tokens({'pad_token': '[PAD]'})


    # eos_token_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)
    # pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

    input_tensor = tokenizer.apply_chat_template(instruction, add_generation_prompt = True, return_tensors='pt')

    input_tensor = input_tensor.to(model.device)

    generation_args = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "do_sample": do_sample,
        "repetition_penalty": repetition_penalty
        # "early_stopping": early_stopping,
        # "eos_token_id": eos_token_id,
        # "pad_token_id": pad_token_id
    }

    print("Successfully loaded model and configs.")
    print("Running inference to generate prompt variations...")
    outputs = model.generate(input_tensor, **generation_args)
    
    # Decode the generated text
    generated_text = tokenizer.decode(outputs[0][input_tensor.shape[1]:], skip_special_tokens=True)
    
    # Enforce the model to start its response with "<think>\n"
    if not generated_text.startswith("<think>\n"):
        generated_text = f"<think>\n{generated_text}"
        
    print("Inference complete.")
    print("Outputs: ", generated_text)
    return generated_text

def parse_model_output(model_output):
    '''
    This function parses the model output to extract the prompt variations as tuples of (bpv_idx, prompt_variation_string).

    Inputs:
        - model_output: The model output string from prompt_variation_inference()

    Outputs:
        - list: A list of tuples, each containing the base prompt variation index and the prompt variation string. Stored as a list of (bpv_idx, bpv_str)
    '''
    print('Model Output:', model_output)

    # PREV CODE USED
    #return [(new_idx, base_prompts[random_idx]) for new_idx, random_idx in enumerate(random_indices)]
    
    if not model_output or model_output.strip() == "":
        raise ValueError("Model output is empty. Check model inference.")

    if isinstance(model_output, dict) and "generated_text" in model_output:
        model_output = model_output["generated_text"]

    json_text = re.search(r"\[.*\]", model_output, re.DOTALL)
    if json_text:
        model_output = json_text.group(0)
    else:
        raise ValueError("No JSON-like output found in model response.")

    return [(idx, pv) for idx, pv in enumerate(json.loads(model_output))]

    # prompt_variations = json.loads(model_output)
    # return [(idx, prompt_variation) for idx, prompt_variation in enumerate(prompt_variations)]

def write_parquet(bp_idx, prompt_variations):
    '''
    Writes the prompt variations to the corresponding parquet file associated with the given base prompt index.

    Inputs:
        - bp_idx: An integer representing the index of the base prompt for which the prompt variations are generated.
        - prompt_variations: A list of tuples containing the prompt variations in the format (bpv_idx, bpv_str).

    Outputs:
        None
    '''

    # # Code to add the base prompt to the list of prompt variations

    # # first, create a BasePromptDB object to get the base prompt string
    # bp_db = BasePromptDB()
    # # then, get the base prompt string using the index
    # bp_str = bp_db.fetch_base_prompt(bp_idx)
    # formatted_bp = [((bp_idx, -1), bp_str)]
    # # then, add the base prompt to the start of the list of prompt variations
    # prompt_variations = formatted_bp + prompt_variations


    pv_parquet = PromptVariationParquet(bp_idx)
    pv_parquet.insert_prompt_variations(prompt_variations)


#collect_instruction(0)

# tester = parse_model_output()
# print(tester)

def main(bp_idx):
    '''
    Runs the entire process of generating prompt variations. It collects the instruction, loads the model, runs inference, and writes the prompt variations to a parquet file.
    '''

    # Step 1: Collect instruction
    instruction = collect_instruction(bp_idx)

    # BECCA: removing bc not needed anymore
    # # Step 2: Load model
    # pipe = load_model()

    # # Step 2: Load model configuration
    configs = load_configs()

    # # Step 3: Run inference to collect prompt variations
    model_output = prompt_variation_inference()

    # tester code

    # model_output = "[\"prompt_variation1\", \"prompt_variation2\", \"prompt_variation3\"]"

    # Step 4: Parse the model output to extract prompt variations
    prompt_variations = parse_model_output(model_output)

    # Step 5: Write the prompt variations to a parquet file
    write_parquet(bp_idx, prompt_variations)



if __name__ == "__main__":
    '''
    Parses the input argument of list of base prompt indices and runs the main function for each index.
    '''
    parser = argparse.ArgumentParser(description="Generate prompt variations for a list of base prompt indices.")
    parser.add_argument(
        "bp_indices",
        type=str,
        help="A list of base prompt indices in the format: [0, 1, 2]"
    )

    args = parser.parse_args()

    try:
        bp_indices = eval(args.bp_indices)
        if not isinstance(bp_indices, list) or not all(isinstance(idx, int) for idx in bp_indices):
            raise ValueError("Invalid format. The argument must be a list of integers in the format: [0, 1, 2]")
    except:
        raise ValueError("The argument must be a list of integers in the format: [0, 1, 2]")

    for bp_idx in bp_indices:
        print(f"Processing base prompt index: {bp_idx}")
        main(bp_idx)
