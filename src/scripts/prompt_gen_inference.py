'''
This script takes a set of base prompts and generates variations of them using a pre-trained model. It then calls regression head to evaluate the generate prompt variations and stores the results.

Process:

- Load the base prompts from a parquet file.
- Load the pre-trained model and tokenizer.
- Construct a prompt structure for the model.
- Generate variations of the base prompts using the pre-trained model.
- Evaluate the generated variations using a regression head model.
- Store the results in a parquet file.
'''


from configs.root_paths import *
from src.utils.data_handler import *
import torch
import argparse
import sys
import optuna
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from peft import get_peft_model, LoraConfig, TaskType
from torch.utils.data import DataLoader
from torch.nn.functional import softmax
from datasets import Dataset
from tqdm import tqdm
import os
import pandas as pd
import json

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_test_base_prompts(start_indices):
    '''
    Returns all the test base prompts

    Args:
        start_indices (list): List of start indices of the test base prompts to be used (example: [500, 800, 900])

    Returns:
        base_prompts (list): List of base prompts
    '''

    # create a data handler bp obj
    # create a list of indices from start_indices: foo([500, 800, 900]) = [500, 501, 502, ..., 599, 800, 801, 802, ..., 899, 900, 901, 902, ..., 999]
    # in data handler make a new function takes a list of indices and returns the base prompts corresponding to those indices
    # call data handler function to get the base prompts
    # return list

    # creating data handler object
    bp_db = BasePromptDB()
    # creating a list of indices from start_indices
    indices = []
    for start_index in start_indices:
        for i in range(start_index, start_index + 100):
            indices.append(i)
    
    # call data handler function to get the base prompts
    list_of_base_prompts = bp_db.fetch_list_of_prompts(indices)

    # return list
    return indices, list_of_base_prompts


def load_test_base_prompts_scores(indices):
    '''
    Returns all the test base prompts validator scores

    Args:
        indices (list): List of start indices of the test base prompts to be used (example: [700, 800, 900])
    
    Returns:
        base_prompt_scores (list): List of base prompt scores
    '''
    # create vs_db object
    # create a new file path for each start index
    # for each file path, retrieve all of the aggregated scores
    # return list

    # file_paths = [os.path.join(VALIDATION_TEST_SCORES, f'{start_index}_validation_score.parquet') for start_index in indices]

    aggregated_scores = []
    for start_index in indices:
        vs_obj = ValidationScoreParquet(start_index, VALIDATION_TEST_SCORES)
        aggregated_scores.extend(vs_obj.fetch_all_agg_validation_scores())

    return aggregated_scores


def load_prompt_gen_model(lora_rank, lora_alpha, dropout_rate):
    """
    Load the base prompt generator model to be fine-tuned with LoRA. Also load the tokenizer.
    Args:
        lora_rank (int): The rank for LoRA.
        lora_alpha (int): The alpha parameter for LoRA.
        dropout_rate (float): The dropout rate for LoRA.
    Returns:
        AutoTokenizer: The tokenizer for the model.
        AutoModelForCausalLM: The prompt generator model with LoRA applied.
    """
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(PROMPT_GEN_BASE_MODEL_ID, use_safetensors=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'left'  # Required for decoder-only models like LLaMA

    # Load the base prompt generator model
    prompt_generator = AutoModelForCausalLM.from_pretrained(
        LORA_PROMPT_GEN_PATH,
        use_safetensors=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # Apply LoRA for fine-tuning
    lora_config = LoraConfig(
        r=lora_rank,  # LoRA attention dimension (rank)
        lora_alpha=lora_alpha,  # Alpha scaling factor for LoRA
        target_modules=["q_proj", "v_proj"],  # Target modules for LoRA
        lora_dropout=dropout_rate,  # Dropout probability for LoRA layers
        task_type=TaskType.CAUSAL_LM  # Task type for causal language modeling
    )
    prompt_generator = get_peft_model(prompt_generator, lora_config)

    # Set the model to training mode
    prompt_generator.train()

    print("Prompt generator model and tokenizer loaded successfully.")
    return tokenizer, prompt_generator

def format_prompt_with_instruction(base_prompt):
    """
    Format the base prompt with an instruction and an example to guide the model in generating a prompt variation.
    Args:
        base_prompt (str): The base prompt to be formatted.
    Returns:
        str: A single formatted input string for the model.
    """

    #test_results: marginally improved by the slightest of margins, Bella identity not destroyed
    example_base_prompt = "Create a learning guide for a third-grader on the concept of gravity."
    example_variation = "Please create a learning guide for a third-grader on the concept of gravity. Thank you so much!"

    instruction = (
            f"Rewrite the following prompt with a polite tone, retaining the same semantics of the base prompt."
            f"The rewritten prompt MUST include the EXACT phrase 'learning guide' and the word 'third-grade' or 'third-grader'."
            f"Do not include any additional text or explanation. Return only the rewritten prompt.\n\n"
            f"Example Base Instruction: {example_base_prompt}\n"
            f"Example Rewritten Prompt: {example_variation}\n\n"
            f"Now, rewrite the following base instruction:\n"
            f"Base Instruction: {base_prompt}\n"
            f"Rewritten Prompt:"
        )

    return instruction

def generate_prompt_variation(prompt_generator, tokenizer, formatted_prompt, max_new_tokens=50):
    """
    Generate a prompt variation using the finetuned prompt generator model.
    Args:
        prompt_generator (AutoModelForCausalLM): The finetuned prompt generator model.
        tokenizer (AutoTokenizer): The tokenizer used for the model.
        formatted_prompt (str): The formatted input string.
        max_new_tokens (int): The maximum number of new tokens to generate.
    Returns:
        str: The generated prompt variation.
    """
    # Tokenize the input
    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    ).to(device)

    # Generate output
    outputs = prompt_generator.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )

    # Decode the generated tokens
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return generated_text

def parse_model_output(generated_text):
    """
    Parse the output from the model to extract the generated prompt variation.
    Args:
        generated_text (str): The output text from the model.
    Returns:
        str: The generated prompt variation.
    """
    # Extract the generated prompt variation from the model output
    # Assuming the output format is consistent and we can split by newlines
    lines = generated_text.split("\n")
    for line in lines:
        if line.startswith("Rewritten Prompt:"):
            #print("Found rewritten prompt line:", line)
            return line.replace("Rewritten Prompt:", "").strip()

def load_regression_head_model():
    """
    Load the regression head model for scoring the generated prompt variations.
    Returns:
        regression_head_model (AutoModelForCausalLM): The regression head model.
        tokenizer (AutoTokenizer): The tokenizer for the regression head model.
    """
    # Load the regression head model
    regression_head_model = AutoModelForCausalLM.from_pretrained(
        BEST_LORA_REGRESSION_HEAD_PATH
    ).to(device)

    regression_head_model.config.output_hidden_states = True

    regression_head_model.regression_head = torch.nn.Linear(
        regression_head_model.config.hidden_size,
        1
    ).to(device).to(torch.float16)
    # Load the regression head weights
    try:
        regression_head_model.regression_head.load_state_dict(torch.load(REGRESSION_HEAD_PATH))
        print("Regression head weights loaded successfully.")
    except Exception as e:
        print(f"Error loading regression head weights: {e}")
        raise
    
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        REGRESSION_HEAD_BASE_MODEL_ID,
        use_safetensors=True
    )

    # Set the pad_token to eos_token if not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return regression_head_model, tokenizer


def call_regression_head(regression_head_model, tokenizer, base_prompt, prompt_variation):
    '''
    Call the regression head model to get the score for the generated prompt variation.
    Args:
        regression_head_model (AutoModelForCausalLM): The regression head model.
        tokenizer (AutoTokenizer): The tokenizer for the regression head model.
        base_prompt (str): The base prompt.
        prompt_variation (str): The generated prompt variation.
    Returns:
        float: The score for the generated prompt variation.
    '''
    combined_input = f'{base_prompt} {prompt_variation}'

    inputs = tokenizer(
        combined_input,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    ).to(device)

    # Get the model's output
    with torch.no_grad():
        outputs = regression_head_model(**inputs)
        hidden_states = outputs.hidden_states[-1]
        pooled_output = hidden_states.mean(dim=1).to(torch.float16).to(device)
        logits = regression_head_model.regression_head(pooled_output)
        prediction = logits.squeeze(-1).cpu().item()

    return prediction


def save_file(indices,base_prompt_strs, base_prompt_scores, bp_regression_scores, pv_str, pv_scores, file_path):
    '''
    Turns the lists into a pandas dataframe and saves it to a csv file in file path
    '''
    df = pd.DataFrame({
        'bp_idx': indices,
        'base_prompt': base_prompt_strs,
        'base_prompt_score': base_prompt_scores,
        'base_prompt_regression_score': bp_regression_scores,
        'generated_pv': pv_str,
        'generated_pv_score': pv_scores
    })
    df.to_csv(file_path, index=False)
    print(f"Results saved to {file_path}")

# def main():
#     '''
#     Collects all Base Prompts, generates variations using the prompt generator model, and evaluates them using the regression head model. Finally, it saves the results to a CSV file.
#     '''
#     # Load the base prompts
#     base_prompts = load_test_base_prompts()

#     # Load the base prompts and their scores
#     base_prompt_scoores = load_test_base_prompts_scores()

#     # Load the prompt generator model and tokenizer
#     tokenizer, prompt_generator = load_prompt_gen_model(lora_rank=16, lora_alpha=32, dropout_rate=0.1)
#     # Load the regression head model and tokenizer
#     regression_head_model, regression_head_tokenizer = load_regression_head_model()

#     prompt_variations = []
#     prompt_variations_scores = []


#     for i in len(base_prompts):
#         base_prompt = base_prompts[i]
#         print(f"Processing base prompt {i+1}/{len(base_prompts)}: {base_prompt}")

#         # Format the base prompt with an instruction
#         formatted_prompt = format_prompt_with_instruction(base_prompt)

#         # Generate a prompt variation
#         generated_text = generate_prompt_variation(prompt_generator, tokenizer, formatted_prompt)
#         prompt_variation = parse_model_output(generated_text)
#         print(f"Generated prompt variation: {prompt_variation}")

#         # Append the generated prompt variation to the list
#         prompt_variations.append(prompt_variation)

#         # Call the regression head model to get the score for the generated prompt variation
#         score = call_regression_head(regression_head_model, regression_head_tokenizer, base_prompt, prompt_variation)
        
#         # Append the score to the list
#         prompt_variations_scores.append(score)

#     df = pd.DataFrame({
#         'base_prompt': base_prompts,
#         'base_prompt_score': base_prompt_scoores,
#         'generated_prompt': prompt_variations,
#         'score': prompt_variations_scores
#     })

#     # PLACEHOLDER - TO BE WORKED ON. WANT TO RETURN LIST OF BASE PROMPTS
#     file_path = os.path.join(MODEL_OUTPUTS, 'prompt_variations_scores.csv')

#     return


def main(file_path):
    '''
    Process:
    1. Load the prompt generator model and tokenizer
    2. Load the regression head model and tokenizer
    3. Load the base prompts and their scores
    4. For each base prompt:
        a. Format the base prompt with an instruction
        b. Generate a prompt variation
        c. Call the regression head model to get the score for the generated prompt variation
        d. Append the generated prompt variation and its score to the list
    5. Aggregate the results and create a csv save file path
    6. Save the results to a CSV file
    '''

    # Process the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('start_indices', type = str)
    args = parser.parse_args()
    list_of_start_indices = eval(args.start_indices)

    # Load the prompt generator model and tokenizer
    tokenizer, prompt_generator = load_prompt_gen_model(lora_rank=16, lora_alpha=32, dropout_rate=0.1)

    # Load the regression head model and tokenizer
    regression_head_model, regression_head_tokenizer = load_regression_head_model()

    indices, base_prompts = load_test_base_prompts(list_of_start_indices)
    base_prompt_scores = load_test_base_prompts_scores(list_of_start_indices)

    base_prompt_regression_scores = []

    prompt_variation_strs = []
    prompt_variations_scores = []

    if len(base_prompts) != len(base_prompt_scores):
        raise ValueError('Base prompts and their scores do not match in length')

    for base_prompt in base_prompts:
        #print(f"Processing base prompt: {base_prompt}")
        # Format the base prompt with an instruction
        formatted_prompt = format_prompt_with_instruction(base_prompt)

        # Generate a prompt variation
        generated_text = generate_prompt_variation(prompt_generator, tokenizer, formatted_prompt)
        prompt_variation = parse_model_output(generated_text)
        #print(f"Generated prompt variation: {prompt_variation}")

        # Call the regression head model to get the score for the generated prompt variation
        pv_score = call_regression_head(regression_head_model, regression_head_tokenizer, base_prompt, prompt_variation)

        bp_regression_score = call_regression_head(regression_head_model, regression_head_tokenizer, base_prompt, base_prompt)
        base_prompt_regression_scores.append(bp_regression_score)
        
        # Append the score to the list
        prompt_variation_strs.append(prompt_variation)
        prompt_variations_scores.append(pv_score)

    # Save the results to a CSV file
    save_file(indices, base_prompts, base_prompt_scores, base_prompt_regression_scores,prompt_variation_strs, prompt_variations_scores, file_path)


if __name__ == "__main__":

    # TO BE WORKED ON: take in start and end index corresponding to the test base prompts to be used
    #main()

    # parser = argparse.ArgumentParser()

    # parser.add_argument('start_indices', type = str)

    # args = parser.parse_args()

    # list_of_start_indices = eval(args.start_indices)


    # # if len(sys.argv) != 1:
    # #     raise ValueError('Usage: python prompt_gen_inference.py <[list of start indices]>')

    # # list_of_start_indices = list(sys.argv[1])

    # # if type(list_of_start_indices) != list:
    # #     raise ValueError('Start indices should be a list of integers')

    # # if all(isinstance(x, int) for x in list_of_start_indices):
    # #     raise ValueError('All start indices should be integers')

    # # testing retrieval of base prompts and their scores

    # base_prompts = load_test_base_prompts(list_of_start_indices)
    # base_prompt_scores = load_test_base_prompts_scores(list_of_start_indices)

    file_path = os.path.join(ANALYSIS_PATH, 'politeness.csv')
    main(file_path)





