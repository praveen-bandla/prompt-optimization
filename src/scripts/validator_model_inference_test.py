"""
Validator Model Inference for Test

This script contains code to automate the procedure of running validator model inference on base prompts.
It takes as input a list of base prompts, generate validator model output accordingly with multiple validator models, and store into corresponding Parquet file.
The score the models assign are based on predefined rubric sections.

Inputs:
- start_idx (int): The starting base prompt index. This is the first base prompt index for which inference will be run.
- end_idx (int): The ending base prompt index. The script runs on all base prompt indices up to end_idx, excluding end_idx itself.

Example usage:
    # Run inference on all base prompts for base prompts 1 to 5.
    python validator_model_inference.py 1 5

Outputs:
- Writes validation scores to a Parquet file named `{i}_validation_score.parquet`, where `i` is the index
  of the base prompt.
"""

from src.utils.prompt import *
from src.utils.data_handler import *
from configs.root_paths import *
import yaml
import json
import os
import torch
import sys
import yaml
from copy import deepcopy

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from huggingface_hub import login

login(token='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')

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

def parse_model_output(model_output):
    '''
    Model output is a list of integers returned as a string. This function parses the string and returns a list of integers.
    '''
    # parsed_list = json.loads(model_output)

    # print(parsed_list)

    # return parsed_list

    try:
        parsed_list = json.loads(model_output)
        return parsed_list
    except (json.JSONDecodeError, TypeError):
        return []

def load_configs():
    '''
    Loads the model configuration file.
    '''
    config_path = VALIDATOR_MODEL_CONFIGS

    with open(config_path, 'r') as file:
      configs = yaml.safe_load(file)

    return configs

def load_models():
    """
    Loads all validator models and their tokenizers.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("CUDA is not available. Please check your setup.")

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

def construct_model_input(base_prompt_str, main_model_output_str):
    '''
    Constructs the model input by replacing placeholders in the val_model_input.json template.

    Args:
    - base_prompt_str (str): The base prompt string.
    - main_model_output_str (str): The main model output string for the given bp_idx.
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
        main_model_output=main_model_output_str, 
        rubric_text=rubric_text
    )

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


def validator_model_inference_per_bp(base_prompt_str, main_model_output_str, models_dict, configs):
    '''
    Runs inference on all three validator models to generate the validation scores for a single base prompt.
    '''
    
    # Step 1: Retrieve global prompt information
    system_role, content = construct_model_input(base_prompt_str, main_model_output_str)

    # Step 2b: Initialize validation scores
    scores = generate_empty_scores_dict()

    # Step 3: For each model, run inference
    for idx, model_info in models_dict.items():
        # Step 3a: load model specific hyperparameters from configs
        model_configs = configs.get(str(model_info['model_name']))

        generation_args = {
            "temperature": model_configs.get("temperature", 0.7),
            "top_p": model_configs.get("top_p", 0.9),
            "top_k": model_configs.get("top_k", 50),
            "max_new_tokens": model_configs.get("max_new_tokens", 100),
            "do_sample": model_configs.get("do_sample", True)
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
            return_full_text=False
        )

        # Step 3d: Run inference
        print(f"DEBUG: Full prompt: {full_prompt}")
        print(f"DEBUG: Generation args: {generation_args}")
        output = pipe(full_prompt, **generation_args)
        print(f"DEBUG: Raw model output: {output}")

        if not output or not output[0].get('generated_text'):
            print(f"WARNING: Model failed to generate output for the prompt. Using default scores.")
            return generate_empty_scores_dict()

        generated_text = output[0]['generated_text']
        print(f"Generated text: {generated_text}")

        # Step 3e: Parse and store the model output
        parsed_list = parse_model_output(generated_text)

        counter = 0
        while len(parsed_list) != NUM_RUBRIC_SECTIONS:
            counter+=1
            print(f'error caught with model generation')
            output = pipe(full_prompt, **generation_args)
            generated_text = output[0]['generated_text']
            parsed_list = parse_model_output(generated_text)

            if counter == 5:
                parsed_list = [0] * NUM_RUBRIC_SECTIONS

        for i, score in enumerate(parsed_list):
            scores[f'section_{i+1}'][idx] = score

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


def validator_model_inference_for_bps(model_dict, configs, mo_parquet, vs_parquet, bp_idx):
    # add vs_parquet as an argument
    '''
    Performs main model inference on all base prompts between bp_idx_start and bp_idx_end. Stores all the outputs in its respective Parquet file.
    '''

    # Load the base prompt data handler
    bp_db = BasePromptDB()

    bp_obj = BasePrompt(bp_idx, bp_db)
    base_prompt_str = bp_obj.get_prompt_str()

    bp_idx_file = math.floor(bp_idx / 100) * 100
    print(bp_idx_file)

    model_output_obj = MainModelOutput(bp_idx=bp_idx_file, mo_parquet=mo_parquet)
    main_model_output_str = model_output_obj.get_output_str_test(bp_idx)

    scores = validator_model_inference_per_bp(base_prompt_str, main_model_output_str, model_dict, configs)

    vs_obj = ValidationScore(bp_idx=bp_idx, vs_parquet=vs_parquet, scores=scores)
    print(f"DEBUG: Validation scores for base prompt {bp_idx}: {scores}")

    return vs_obj


def write_validation_scores_to_parquet(vs_parquet, vs_objs):
    # add vs_parquet as an argument
    '''
    Writes the validation scores to a Parquet file.

    Args:
    - vs_objs (list): List of ValidationScore objects.
    '''
    vs_parquet.insert_validation_scores_test(vs_objs)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise ValueError("Please provide a list of base prompt indices.")
    
    start_idx = int(sys.argv[1])
    end_idx = int(sys.argv[2])
    models_dict = load_models()
    configs = load_configs()

    all_vs = []
    # Run inference for the given base prompt index
    mo_parquet = ModelOutputParquet(start_idx, MODEL_TEST_OUTPUTS)
    vs_parquet = ValidationScoreParquet(bp_idx=start_idx, test=True, parquet_root_path=VALIDATION_TEST_SCORES)
    for bp_idx in range(start_idx, end_idx):
        all_vs.append(validator_model_inference_for_bps(models_dict, configs, mo_parquet, vs_parquet, bp_idx))
    
    # Write the validation scores to Parquet
    write_validation_scores_to_parquet(vs_parquet, all_vs)
    
    print("All inferences have been completed.")