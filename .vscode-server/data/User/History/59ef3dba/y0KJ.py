#!/usr/bin/env python3
'''
Generate Base Prompts (New)

This updated script generates a set of base prompts by calling the main model to produce base prompts based on the stored inputs. It runs inferences and generates 'n' base prompts for further use (n found in the data_size_configs file).

Inputs:
    None

Dependencies:
    - base_prompt_model_input.json: A JSON file containing instructions for generating base prompts.
    - base_prompt_model_config.yaml: A YAML file specifying model parameters and settings.
    - data_handler.py: base prompt class
    - prompt.py: base prompt class

Outputs:
    None

Process:
1. Collect the instruction for generating base prompts. Uses the number_of_base_prompts from the data_size_configs file for n and the base_prompt_model_input.json file for the instruction.
2. Loads the model
3. Loads the model configuration file
4. Runs inference to collect all the base prompts
5. Writes the base prompts to a SQLite database

'''

from src.utils.prompt import *
from src.utils.data_handler import *
from configs.root_paths import *
import yaml
import json
import random
import torch
import transformers
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from huggingface_hub import login

login(token="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")

# Step 1: Collect the instruction for generating base prompts
def collect_instruction():
    '''
    Creates instructions for generating base prompts. Uses the base_prompt_model_input.json file as the model input, along with NUM_BASE_PROMPTS from the data_size_configs file.

    Inputs: None

    Outputs:
        - full_prompt: A list containing the system role and content template for generating base prompts, stored as two dictionaries.
    
    '''
    if not os.path.exists(BASE_PROMPT_MODEL_INPUT):
        raise FileNotFoundError(f"Instruction file not found at {BASE_PROMPT_MODEL_INPUT}")
    with open(BASE_PROMPT_MODEL_INPUT, 'r') as f:
        prompt_structure = json.load(f)
    
    system_role = prompt_structure["system_role"]
    content_template = prompt_structure["content_template"]
    #instruction = system_role + " " + content_template

    content = content_template.format(num_prompt = NUM_BASE_PROMPTS)

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
        - configs: the configurations for the base prompt model.
    '''
    config_path = BASE_PROMPT_MODEL_CONFIG

    with open(config_path, 'r') as f:
        configs = yaml.safe_load(f)
      
    return configs

# # Define manual formatting function
# def format_chat(system_prompt, user_prompt):
#     """
#     Manually format the chat if `chat_template` is missing.
#     """
#     return f"<|system|>\n{system_prompt}\n\n<|user|>\n{user_prompt}\n\n<|assistant|>\n"

def load_model():
    '''
    Loads the base prompt model for inference. 
    '''

    # testing using model_id
    base_prompt_model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path = base_prompt_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
        #local_files_only=True
    )

    #tokenizer = AutoTokenizer.from_pretrained(BASE_PROMPT_MODEL, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(base_prompt_model_id, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    
    # Load Configuration JSON
    # with open(BASE_PROMPT_MODEL_INPUT, "r") as file:
    #     input_data = json.load(file)

    # system_role, content = collect_instruction()

    # formatted_chat = format_chat(system_role, content)

    # print("Formatted Chat Input:\n", formatted_chat)

    pipe = pipeline(
        'text-generation', 
        model=model, 
        tokenizer=tokenizer, 
        device_map="auto"
    )

    return model, tokenizer

    # model_id = "meta-llama/Llama-3.1-8B"
    # tokenizer = AutoTokenizer.from_pretrained(model_id)
    # tokenizer.chat_template = "<s>[INST] {instruction} [/INST] {response}</s>"  # Example template

    # pipeline = transformers.pipeline(
    #     "text-generation",
    #     model=model_id,
    #     tokenizer=tokenizer,
    #     model_kwargs={"torch_dtype": torch.bfloat16},
    #     device_map="auto",
    # )

    # return pipeline
    

# Step 2: Run inference to collect all the base prompts

def base_prompt_inference(): 
    '''
    This runs inference on the base_prompt_model to generate the desired output. It solely retrieves the response as a string and does not process it further.
    '''
    instruction = collect_instruction()
    model, tokenizer = load_model()
    configs = load_configs()

    max_new_tokens = configs.get("max_new_tokens")
    temperature = configs.get("temperature")
    top_p = configs.get("top_p")
    top_k = configs.get("top_k")
    do_sample = configs.get("do_sample")
    repetition_penalty = configs.get("repetition_penalty")
    print("instruction :", instruction)

    input_tensor = tokenizer.apply_chat_template(instruction, add_generation_prompt = True, return_tensors='pt')

    input_tensor = input_tensor.to(model.device)

    generation_args = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "repetition_penalty": repetition_penalty,
        "do_sample": do_sample
    }

    print("Successfully loaded model and configs.")
    print("Running inference to generate prompt variations...")
    #outputs = pipe(instruction, **generation_args)
    outputs = model.generate(input_tensor, **generation_args)
    generated_text = tokenizer.decode(outputs[0][input_tensor.shape[1]:], skip_special_tokens=True)
    print("Outputs: ", generated_text)
    
    # Extract the portion after </think>
    try:
        if "</think>" in generated_text:
            generated_text = generated_text.split("</think>", 1)[1].strip()  # Get everything after </think>
        
        # Use regex to extract the JSON object containing the prompts
        json_match = re.search(r"{(.|\n)*?}", generated_text, re.DOTALL)
        if not json_match:
            raise ValueError("No JSON object found in the model output.")
        
        json_object = json.loads(json_match.group(0))  # Parse the JSON object
        prompts = json_object.get("prompts")  # Extract the 'prompts' key
        if not prompts:
            raise ValueError("No 'prompts' key found in the JSON object.")
        
        print("Extracted Prompts:", prompts)
        return prompts
    except Exception as e:
        print("Error extracting prompts:", e)
        raise

    # Convert the instruction (list of dictionaries) into a formatted string
    system_prompt = instruction[0]["content"]
    user_prompt = instruction[1]["content"]
    input_text = tokenizer.apply_chat_template(system_prompt, user_prompt)

    generation_args = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "repetition_penalty": repetition_penalty,
        "do_sample": do_sample
    }

    print("Successfully loaded model and configs.")
    print("Running inference to generate base prompts...")
    # Run inference with the formatted chat input
    outputs = pipe(input_text, **generation_args)

    print("Raw Model Output:", outputs)
    generated_text = outputs[0]["generated_text"]
    
    print("Inference complete.")
    print("Outputs: ", generated_text)
    return generated_text

    # # Extract the assistant's response from the generated text
    # assistant_response = None
    # if isinstance(outputs, list) and "generated_text" in outputs[0]:
    #     generated_text = outputs[0]["generated_text"]

    #     # If the model outputs a list of responses, extract the assistant's part
    #     if isinstance(generated_text, str):  # Handle direct string outputs
    #         assistant_response = generated_text.split("<|assistant|>\n")[-1].strip()
    #     elif isinstance(generated_text, list):  # Handle list outputs
    #         for item in generated_text:
    #             if item.get("role") == "assistant":
    #                 assistant_response = item.get("content")
    #                 break

    # if not assistant_response:
    #     raise ValueError("No assistant response found in the model output.")

    # print("Inference complete.")
    # print("Assistant Response:\n", assistant_response)
    
    # return assistant_response

def parse_model_output_as_bp_objects(model_output):
    '''
    This function parses the model output to extract the base prompts as tuples of (bp_idx, base_prompt_string). NB: This assumes that the db is empty. This handles the randomization of ordering of prompts as well.

    Inputs:
        - model_output: The model output string from base_prompt_inference()

    Outputs:
        - list: A list of tuples, each containing the base prompt index and the base prompt string. Stored as a list of (bp_idx, bp_str)
    '''
    base_prompts = json.loads(model_output)

    # creating random order of prompts stored as int indices
    random_indices = random.sample(range(NUM_BASE_PROMPTS), NUM_BASE_PROMPTS)

    # returning a list of tuples in the desired format of the random order of prompts
    return [(new_idx, base_prompts[random_idx]) for new_idx, random_idx in enumerate(random_indices)]


# Step 3: Write the base prompts to a SQLite database
def write_to_db(formatted_base_prompts, bp_db):
    '''
    This function writes the base prompts to the SQLite database using the BasePromptDB object.

    Inputs:
        - formatted_base_prompts: A list of tuples, each containing the base prompt index and the base prompt string. Stored as a list of (bp_idx, bp_str)
        - bp_db: an object of type BasePromptDB (data handler) used to write
    '''
    bp_db.insert_base_prompts(formatted_base_prompts)

    return


def main():
    if os.path.exists(SQL_DB): #ensures that we do not have a pre-existing database
        os.remove(SQL_DB)
    # creates the BasePromptDB object
    bp_db = BasePromptDB()

    # generates model output by inferencing
    model_output = base_prompt_inference()
    
    # formats the model output as a list of tuples in random order
    formatted_base_prompts = parse_model_output_as_bp_objects(model_output)

    #formatted_base_prompts = [(1, 'testing1'), (2, 'testing2'), (3, 'tesingt3'), (4, 'test4'), (5, 'test5')]
    
    # writes the base prompts to the SQLite database
    write_to_db(formatted_base_prompts, bp_db)
    # closes the connection to the database
    bp_db.close_connection()
    print("Base prompts generated and written to SQLite database.")


if __name__ == "__main__":
    main()
    # pass
    # bp_db = BasePromptDB()
    # bp_db.reset_database()