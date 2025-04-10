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
import yaml
import json
import os
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from mistral_inference.transformer import Transformer
from mistral_inference.generate import generate

from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest


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

    falcon_tokenizer = AutoTokenizer.from_pretrained(FALCON_MAMBA_MODEL_ID)
    falcon_model = AutoModelForCausalLM.from_pretrained(FALCON_MAMBA_MODEL_ID).to(device)
    
    opt_tokenizer = AutoTokenizer.from_pretrained(OPT_MODEL_ID)
    opt_model = AutoModelForCausalLM.from_pretrained(OPT_MODEL_ID).to(device)

    mistral_tokenizer = AutoTokenizer.from_pretrained(MISTRAL_INSTRUCT_MODEL_ID)
    mistral_model = AutoModelForCausalLM.from_pretrained(MISTRAL_INSTRUCT_MODEL_ID).to(device)      
      
    return {
        'falcon-mamba': (falcon_model, falcon_tokenizer),
        'opt': (opt_model, opt_tokenizer),
        'mistral': (mistral_model, mistral_tokenizer)
    }


def construct_model_input(bpv_idx):
    '''
    Constructs the model input by replacing placeholders in the val_model_input.json template.
    '''

    # Collect base prompt string from bpv_idx
    bp_idx = (bpv_idx[0], 1)
    base_prompt_data_handler = BasePrompt(bp_idx)
    base_prompt_str = base_prompt_data_handler.get_prompt_str()

    # Fetch the main model output string for the given bpv_idx
    mo_parquet = ModelOutputParquet(bp_idx[0])
    main_model_output_str = mo_parquet.fetch_model_output(bpv_idx)

    if not main_model_output_str:
        raise ValueError(f"No main model output found for bpv_idx {bpv_idx}")

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

def validator_model_inference(bpv_idx, models):
    '''
    Runs inference on all three validator models to generate the validation scores for a single bpv_idx.
    '''
    instruction = construct_model_input(bpv_idx)
    model, tokenizer = load_models()
    configs = load_configs()

    # Retrieve the hyperparameters
    temperature = configs.get("temperature")
    top_p = configs.get("top_p")
    top_k = configs.get("top_k")

    generation_args = {
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
    }

    # Initialize validation scores
    validation_scores = ValidationScore(bpv_idx, ValidationScoreParquet(bpv_idx[0]))

    # Run inference sequentially for each model
    for model_name, (model, tokenizer) in models.items():
        print(f'Running inference for model: {model_name}')
        
        # Create a pipeline for the current model
        pipe = pipeline(
            'text-generation', 
            model=model, 
            tokenizer=tokenizer,
            device=0  # Use GPU
        )
        output = pipe(instruction, **generation_args)

        # Parse and store the model output
        generated_text = output[0]['generated_text']
        parsed_list = validation_scores.parse_model_output(generated_text)
        validation_scores.append_model_scores(parsed_list)

    # Store scores by section and calculate aggregated scores
    validation_scores.store_scores_by_section(validation_scores.scores)
    validation_scores.store_aggregated_scores()

    return validation_scores.scores


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


def main(start_idx, end_idx):
    """
    Main function to process all prompt variations for a given range of indices.

    Args:
    - start_idx (tuple): The starting base prompt variation index (e.g., (1, 0)).
    - end_idx (tuple): The ending base prompt variation index (e.g., (1, 4)).
    """
    # Load model configurations
    configs = load_configs()

    # Load all models
    models = load_models()

    # Collect all scores for all prompt variations
    all_scores = []
    for bpv_idx in range(start_idx[1], end_idx[1]):
        bpv_idx_tuple = (start_idx[0], bpv_idx)
        print(f"Processing bpv_idx: {bpv_idx_tuple}")
        scores = validator_model_inference(bpv_idx_tuple, models)
        all_scores.append(scores)

    # Write all scores to Parquet
    vs_parquet = ValidationScoreParquet(start_idx[0])
    vs_parquet.insert_validation_scores(all_scores)

    print(f"Validation scores for base prompt {start_idx[0]} have been saved to the Parquet file.")


if __name__ == "__main__":
    import sys

    # Parse command-line arguments
    if len(sys.argv) != 3:
        print("Usage: python validator_model_inference.py <start_idx> <end_idx>")
        print("Example: python validator_model_inference.py '(1, 0)' '(1, 4)'")
        sys.exit(1)

    # Convert string arguments to tuples
    start_idx = eval(sys.argv[1])  # Example: (1, 0)
    end_idx = eval(sys.argv[2])    # Example: (1, 4)

    # Run the main function
    main(start_idx, end_idx)