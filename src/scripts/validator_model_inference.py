"""
Validator Model Inference

This script contains code to automate the procedure of running validator model inference on prompt variations.
It takes as input a list of bpv_idx, generate validator model output accordingly with multiple validator models, and store into corresponding Parquet file.
The score the models assign are based on predefined rubric sections.

Inputs:
- start_idx (int): The starting prompt variation index. This is the first base prompt index for which inference will be run.
- end_idx (int): The ending prompt variation index. The script runs on all base prompt indices up to end_idx, excluding end_idx itself.

Example usage:
    # Run inference on all prompt variations for prompt variation indices 0,1,2,3,4 of base prompt 1.
    python validator_model_inference.py (1, 0) (1, 4)

Outputs:
- Writes validation scores to a Parquet file named `{i}_validation_score.parquet`, where `i` is the index
  of the base prompt.
- The output structure consists of **2n + 1 columns**, with n rubric sections. The last column is the total score, weighted.
- The first n columns contain a **tuple of k values**, corresponding to scores assigned by the `k` number of validator models.
Example format of a single file, if n = 3 and k = 3:
| bpv_idx | section_1 | section_2 | section_3 | section_1_avg | section_2_avg | section_3_avg | total_score |
|---------|-----------|-----------|-----------|---------------|---------------|---------------|-------------|
| (1, 0)  | (2, 1, 3) | (1, 2, 3) | (3, 1, 2) | 2.0           | 2.0           | 2.0           | 4.0         |
| (1, 1)  | (3, 2, 1) | (2, 1, 3) | (1, 3, 2) | 2.0           | 2.0           | 2.0           | 4.0         |
| ...     | ...       | ...       | ...       | ...           | ...           | ...           | ...         |

Process:
1. Collects the instruction for generating validator scores, using the base prompt string and the main model output strings.
2. Loads all validator models
3. Loads the model configuration file
4. Runs inference on all validator models to collect all validator scores
5. Opens or creates the corresponding Parquet file and writes results to it.

Dependencies:
- `val_model_input.json`: A JSON file containing instructions for validation.
- `validator_model_config.yaml`: A YAML configuration file specifying parameters for the validator models.
"""

from src.utils.prompt import *
from src.utils.data_handler import *
from configs.root_paths import *
from configs.data_size_configs import MODEL_ALIASES
import yaml
import json
import os
import torch
import sys
import yaml

# adding
from copy import deepcopy

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
# from mistral_inference.transformer import Transformer
# from mistral_inference.generate import generate

# from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
# from mistral_common.protocol.instruct.messages import UserMessage
# from mistral_common.protocol.instruct.request import ChatCompletionRequest


# PRAVEEN: helper function to generate empty scores dict
def generate_empty_scores_dict(num_sections = NUM_RUBRIC_SECTIONS, num_models = NUM_VALIDATOR_MODELS):
    blank_dict = {}

    # Add section scores as tuples of 0
    for i in range(1, num_sections + 1):
        blank_dict[f"section_{i}"] = [0] * num_models

    # Add section averages as 0
    for i in range(1, num_sections + 1):
        blank_dict[f"section_{i}_avg"] = 0

    # Add total score as 0
    blank_dict["total_score"] = 0

    return blank_dict

# PRAVEEN: helper function to parse through model output
def parse_model_output(model_output):
    '''
    Model output is a list of integers returned as a string. This function parses the string and returns a list of integers.
    '''
    # # Split the string by commas and convert to integers
    # parsed_list = [int(x.strip()) for x in model_output.split(',') if x.strip().isdigit()]
    parsed_list = json.loads(model_output)

    print(parsed_list)

    return parsed_list

def load_configs():
    '''
    Loads the model configuration file.
    '''
    config_path = VALIDATOR_MODEL_CONFIGS

    with open(config_path, 'r') as file:
      configs = yaml.safe_load(file)

    return configs

def load_models():
    """Loads all validator models and their tokenizers."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("CUDA is not available. Please check your setup.")

    # Map model aliases to their respective models and tokenizers
    # PRAVEEN: RE-IMPLEMENTING WITH NEW DICT AND VARIABLE NAMES
    # model_dict = {}
    # for alias, model_name in MODEL_ALIASES.items():
    #     tokenizer = AutoTokenizer.from_pretrained(globals()[f'{model_name.upper()}_MODEL_ID'])
    #     model = AutoModelForCausalLM.from_pretrained(globals()[f'{model_name.upper()}_MODEL_ID']).to(device)
    #     model_dict[alias] = (model, tokenizer)


    # PRAVEEN: NEW IMPLEMENTATION
    models_dict = deepcopy(VAL_MODEL_DICT)

    for key, model_info in models_dict.items():
        print('Starting to load model:', model_info['model_name'])
        model_id = model_info['huggingface_model_id']

        print(f'model_id: {model_id}')
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )

        # Update the dictionary with the loaded model and tokenizer
        models_dict[key] = {
            'model_name': model_info['model_name'],
            'huggingface_model_id': model_info['huggingface_model_id'],
            'prompt_structure': model_info['prompt_structure'],
            'model': model,
            'tokenizer': tokenizer
        }
        print('Finished loading model:', model_info['model_name'])
    
    return models_dict

    # falcon_tokenizer = AutoTokenizer.from_pretrained(FALCON_MAMBA_MODEL_ID)
    # falcon_model = AutoModelForCausalLM.from_pretrained(FALCON_MAMBA_MODEL_ID).to(device)
    
    # opt_tokenizer = AutoTokenizer.from_pretrained(OPT_MODEL_ID)
    # opt_model = AutoModelForCausalLM.from_pretrained(OPT_MODEL_ID).to(device)

    # mistral_tokenizer = AutoTokenizer.from_pretrained(MISTRAL_INSTRUCT_MODEL_ID)
    # mistral_model = AutoModelForCausalLM.from_pretrained(MISTRAL_INSTRUCT_MODEL_ID).to(device)      
      
    # return {
    #     'falcon-mamba': (falcon_model, falcon_tokenizer),
    #     'opt': (opt_model, opt_tokenizer),
    #     'mistral': (mistral_model, mistral_tokenizer)
    # }


def construct_model_input(base_prompt_str, main_model_output_str):
    '''
    Constructs the model input by replacing placeholders in the val_model_input.json template.

    Args:
    - base_prompt_str (str): The base prompt string.
    - main_model_output_str (str): The main model output string for the given bpv_idx.
    '''

    # Load the rubric text from the rubric.txt file
    if not os.path.exists(RUBRIC_PATH):
        raise FileNotFoundError(f"Rubric file not found at {RUBRIC_PATH}")
    with open(RUBRIC_PATH, 'r') as file:
        rubric_text = file.read()

    # Load the input model template
    if not os.path.exists(VALIDATOR_MODEL_INPUT):
        raise FileNotFoundError(f"Validator model input file not found at {VALIDATOR_MODEL_INPUT}")
    with open(VALIDATOR_MODEL_INPUT, 'r') as file:
        model_input = json.load(file)

    system_role = model_input['system_role']
    content_template = model_input['content_template']

    # Replace placeholders with actual values
    content = content_template.format(
        base_prompt_str=base_prompt_str, 
        main_model_output =main_model_output_str, 
        rubric_text = rubric_text
    )
       
    # BECCA: Removing because not all models may consider system_role and content separately
    # e.g. Deepseek combines both into content
    # full_prompt = [
    #     {
    #         "role": "system",
    #         "content": system_role
    #     },
    #     {
    #         "role": "user",
    #         "content": content
    #     } 
    # ]

    # return full_prompt

    return system_role, content

def construct_prompt(prompt_structure, system_role, content):
    '''
    Based on the prompt structure, constructs the full prompt for the model provided the information
    '''
    if prompt_structure == [
            {
                "role": "user",
                "content": ""
            }
        ]:
        return [
            {
                "role": "user",
                "content": f'{system_role} {content}'
            }
        ]
    elif prompt_structure == [
            {
                "role": "system",
                "content": ""
            },
            {
                "role": "user",
                "content": ""
            }
        ]:
        return [
            {
                "role": "system",
                "content": system_role
            },
            {
                "role": "user",
                "content": content
            }
        ]


# PRAVEEN: I COMMENTED OUT TO CHANGE FUNCTION IMPLEMENTATION AND NAME
# def validator_model_inference(bpv_idx, models, vs_parquet):

#     # Construct full prompt based on model-specific needs
#     '''
#     Runs inference on all three validator models to generate the validation scores for a single bpv_idx.
#     '''
#     system_role, content = construct_model_input(bpv_idx)

#     for model in model_dict.items():
#         # return 

#     model_dict = load_models()
#     configs = load_configs()

#     # Retrieve the hyperparameters
#     temperature = configs.get("temperature")
#     top_p = configs.get("top_p")
#     top_k = configs.get("top_k")

#     generation_args = {
#         "temperature": temperature,
#         "top_p": top_p,
#         "top_k": top_k,
#     }

#     # Initialize validation scores
#     validation_scores = ValidationScore(bpv_idx, vs_parquet)

#     # Run inference sequentially for each model
#     for model_name, (model, tokenizer) in model_dict.items():
#         print(f'Running inference for model: {model_name}')

        
#         # Create a pipeline for the current model
#         pipe = pipeline(
#             'text-generation', 
#             model=model, 
#             tokenizer=tokenizer,
#             device=0  # Use GPU
#         )
#         output = pipe(instruction, **generation_args)

#         # Parse and store the model output
#         generated_text = output[0]['generated_text']
#         parsed_list = validation_scores.parse_model_output(generated_text)
#         validation_scores.append_model_scores(parsed_list)

#     # Store scores by section and calculate aggregated scores
#     validation_scores.store_scores_by_section(validation_scores.scores)
#     validation_scores.store_aggregated_scores()

#     return validation_scores.scores


def validator_model_inference_per_prompt_variation(base_prompt_str, main_model_output_str, models_dict, configs):
    '''
    Runs inference on all three validator models to generate the validation scores for a single bpv_idx.
    '''
    
    # Step 1: Retrieve global prompt information
    system_role, content = construct_model_input(base_prompt_str, main_model_output_str)

    # # Step 2a: Load models_dict and configs
    # models_dict = load_models()
    # configs = load_configs()
    # print(configs)

    # Step 2b: Initialize validation scores
    scores = generate_empty_scores_dict()

    # Step 3: For each model, run inference
    for idx, model_info in models_dict.items():
        # Step 3a: load model specific hyperparameters from configs

        model_configs = configs.get(str(model_info['model_name']))

        generation_args = {
            "temperature": model_configs.get("temperature"),
            "top_p": model_configs.get("top_p"),
            "top_k": model_configs.get("top_k"),
            "max_new_tokens": model_configs.get("max_new_tokens"),
            "do_sample": model_configs.get("do_sample")
        }

        # Step 3b: Construct the prompt based on the model's requirements
        prompt_structure = model_info['prompt_structure']
        full_prompt = construct_prompt(prompt_structure, system_role, content)

        # Step 3c: Create a pipeline for the current model
        print(f'Running inference for model: {model_info["model_name"]}')
        pipe = pipeline(
            'text-generation', 
            model=models_dict[idx]['model'], 
            tokenizer=models_dict[idx]['tokenizer'],
            device_map="auto",  # Use GPU
            return_full_text = False
        )

        # Step 3d: Run inference
        output = pipe(full_prompt, **generation_args)
        generated_text = output[0]['generated_text']

        print(f"Generated text: {generated_text}")

        # Step 3e: Parse and store the model output
        parsed_list = parse_model_output(generated_text)

        if len(parsed_list) != NUM_RUBRIC_SECTIONS:
            raise ValueError(f"Parsed list length {len(parsed_list)} does not match expected number of sections {NUM_RUBRIC_SECTIONS}")
        
        for i, score in enumerate(parsed_list):
            scores[f"section_{i + 1}"][idx] = score
    
    # Step 4: calculate aggregated scores
    for i in range(1, NUM_RUBRIC_SECTIONS + 1):
        section_scores = scores[f"section_{i}"]
        avg_score = sum(section_scores) / len(section_scores)
        scores[f"section_{i}_avg"] = avg_score
    
    # Step 5: Calculate total average score
    total_score = 0
    for i in range(1, NUM_RUBRIC_SECTIONS + 1):
        total_score += scores[f"section_{i}_avg"] * SECTION_WEIGHTS[f'section_{i}']
    scores["total_score"] = round(total_score, 2)

    return scores


def validator_model_inference_per_base_prompt(mo_parquet, vs_parquet, bp_idx):
    # add vs_parquet as an argument
    '''
    Performs main model inference on all prompt variations for the given bp_idx. Stores all the outputs in its respective Parquet file.

    Args:
    - bp_idx (int): The base prompt index
    '''
    all_vs_objs = []
    
    # Load the base prompt data handler
    bp_db = BasePromptDB()

    # Load the base prompt obj
    bp_obj = BasePrompt(bp_idx, bp_db)
    # Load the base_prompt_str
    base_prompt_str = bp_obj.get_prompt_str()

    models_dict = load_models()
    configs = load_configs()

    for idx in range(-1, NUM_PROMPT_VARIATIONS):
        # Load the main model output string for the given bpv_idx
        model_output_obj = MainModelOutput((bp_idx, idx), mo_parquet)
        main_model_output_str = model_output_obj.get_output_str()

        # Run inference
        scores = validator_model_inference_per_prompt_variation(base_prompt_str, main_model_output_str, models_dict, configs)
        #scores = generate_empty_scores_dict()

        # Create a new ValidationScore object
        vs_obj = ValidationScore((bp_idx, idx), vs_parquet, scores)
        all_vs_objs.append(vs_obj)

    return all_vs_objs


def write_validation_scores_to_parquet(vs_parquet, vs_objs, bp_idx):
    # add vs_parquet as an argument
    '''
    Writes the validation scores to a Parquet file.

    Args:
    - vs_objs (list): List of ValidationScore objects.
    - bp_idx (int): The base prompt index.
    '''
    # Create a new Parquet file for the validation scores
    # vs_parquet = ValidationScoreParquet(bp_idx)

    vs_parquet.insert_validation_scores(vs_objs)



# BECCA: Do we still need this function or is it now unncessary because we are storing model outputs for base prompts and prompt variations in the same parquet file?
# def validator_model_inference_per_base_prompt(bp_idx):
#     # Less priority, used in testing pipeline afaik
#     '''
#     Performs main model inference on all prompt variations for the given bp_idx. Stores all the outputs in its respective Parquet file.

#     Input:
#     - bp_idx (int): The base prompt index.
#     '''
#     all_pv_outputs = []
#     mo_parquet = prompt.ModelOutputParquet(bp_idx)
#     for pv_obj in collect_prompt_variations(bp_idx):
#         model_output = validator_model_inference_per_prompt_variation(pv_obj)
#         all_pv_outputs.append((pv_obj.get_prompt_index(), model_output))
#     mo_parquet.insert_model_outputs(all_pv_outputs)


# def main(start_idx, end_idx):
#     '''
#     Main function to process all prompt variations for a given range of indices.

#     Args:
#     - start_idx (tuple): The starting base prompt variation index (e.g., (1, 0)).
#     - end_idx (tuple): The ending base prompt variation index (e.g., (1, 4)).
#     '''
#     # Load model configurations
#     configs = load_configs()

#     # Load all models
#     models = load_models()

#     # Collect base prompt string from bpv_idx
#     bp_idx = (bpv_idx[0])
#     base_prompt_data_handler = BasePrompt(bp_idx)

#     # Fetch the main model output string for the given bpv_idx
#     mo_parquet = ModelOutputParquet(bp_idx[0])

#     construct_model_input(bpv_idx, base_prompt_data_handler, mo_parquet)

#     # Collect all scores for all prompt variations
#     all_scores = []
#     for bpv_idx in range(start_idx[1], end_idx[1]):
#         bpv_idx_tuple = (start_idx[0], bpv_idx)
#         print(f"Processing bpv_idx: {bpv_idx_tuple}")
#         scores = validator_model_inference(bpv_idx_tuple, models)
#         all_scores.append(scores)
#         # Create function for runnign inference per prompt variaiton
#         # inside function for running inference per base prompt, loop through first function
#         # Refer to main_model_inference.py for example

#     # Write all scores to Parquet
#     vs_parquet = ValidationScoreParquet(start_idx[0])
#     vs_parquet.insert_validation_scores(all_scores)

#     print(f"Validation scores for base prompt {start_idx[0]} have been saved to the Parquet file.")


if __name__ == "__main__":
    # import sys

    # # Parse command-line arguments
    # if len(sys.argv) != 3:
    #     print("Usage: python validator_model_inference.py <start_idx> <end_idx>")
    #     print("Example: python validator_model_inference.py '(1, 0)' '(1, 4)'")
    #     sys.exit(1)

    # # Convert string arguments to tuples
    # start_idx = eval(sys.argv[1])  # Example: (1, 0)
    # end_idx = eval(sys.argv[2])    # Example: (1, 4)

    # # Run the main function
    # main(start_idx, end_idx)
    if len(sys.argv) != 3:
        raise ValueError("Please provide a list of base prompt indices.")
    
    start_idx = int(sys.argv[1])
    end_idx = int(sys.argv[2])
    for bp_idx in range(start_idx, end_idx):
        vs_parquet = ValidationScoreParquet(bp_idx)
        # Run inference for the given base prompt index
        mo_parquet = ModelOutputParquet(bp_idx)
        vs_parquet = ValidationScoreParquet(bp_idx)

        vs_objs = validator_model_inference_per_base_prompt(mo_parquet, vs_parquet, bp_idx)
        
        # Write the validation scores to Parquet
        write_validation_scores_to_parquet(vs_parquet, vs_objs, bp_idx)
        
        print(f"Finished inference for base prompt index {bp_idx}.")

    print("All inferences have been completed.")

