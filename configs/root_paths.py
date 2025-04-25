'''
This document contains the paths to the root directories of the project.
'''

import os

# Where the main folder is:
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Specific subfolders
DATA_PATH = os.path.join(ROOT_PATH, 'data')
MODELS_PATH = os.path.join(ROOT_PATH, 'models')
CONFIGS_PATH = os.path.join(ROOT_PATH, 'configs')
MODEL_INPUT_PATH = os.path.join(ROOT_PATH, 'model_input')
MODEL_CONFIGS = os.path.join(CONFIGS_PATH, 'model_configs')
#MODEL_INPUTS = os.path.join(ROOT_PATH, 'model_input')

# Validation model related files
VALIDATION_SCORES = os.path.join(DATA_PATH, 'validation_scores')
VALIDATION_TEST_SCORES = os.path.join(DATA_PATH, 'validation_test_scores')
VALIDATOR_MODEL_INPUT = os.path.join(MODEL_INPUT_PATH, 'validator_model_input.json')

VALIDATOR_MODEL_CONFIGS = os.path.join(MODEL_CONFIGS, 'validator_model_config.yaml')
# FALCON_MAMBA_MODEL = os.path.join(MODELS_PATH, 'tiiuae/falcon-mamba-7b')
# FALCON_MAMBA_MODEL_ID = 'tiiuae/falcon-mamba-7b'
OPT_MODEL = os.path.join(MODELS_PATH, 'facebook/opt-6.7b')
OPT_MODEL_ID = 'facebook/opt-6.7b'
MISTRAL_INSTRUCT_MODEL = os.path.join(MODELS_PATH, 'mistralai/Mistral-7B-Instruct-v0.3')
MISTRAL_INSTRUCT_MODEL_ID = 'mistralai/Mistral-7B-Instruct-v0.3'
RUBRIC_PATH = os.path.join(MODEL_INPUT_PATH, 'rubric.txt')


# BP and BPV related files
SQL_DB = os.path.join(DATA_PATH, 'base_prompts', 'base_prompts.db')
SQL_TOPIC_DB = os.path.join(DATA_PATH, 'base_prompts', 'prompt_topics.db')
PROMPT_VARIATIONS = os.path.join(DATA_PATH, 'prompt_variations')
PROMPT_VARIATION_MODEL = os.path.join(MODELS_PATH, 'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B')
PROMPT_VARIATION_MODEL_ID = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B'
# PROMPT_VARIATION_MODEL = os.path.join(MODELS_PATH, 'prithivMLmods/Qwen2.5-1.5B-DeepSeek-R1-Instruct')
# PROMPT_VARIATION_MODEL_ID = 'prithivMLmods/Qwen2.5-1.5B-DeepSeek-R1-Instruct'

PROMPT_VARIATION_MODEL_INPUT = os.path.join(MODEL_INPUT_PATH, 'prompt_variation_input.json')
PROMPT_VARIATION_MODEL_CONFIG = os.path.join(CONFIGS_PATH, 'model_configs/prompt_variation_config.yaml')
PROMPT_TOPIC_MODEL_INPUT = os.path.join(MODEL_INPUT_PATH, 'base_prompt_topics.json')
BASE_PROMPT_MODEL_INPUT = os.path.join(MODEL_INPUT_PATH, 'base_prompt_input.json')
BASE_PROMPT_MODEL_CONFIG = os.path.join(CONFIGS_PATH, 'model_configs/base_prompt_config.yaml')
PROMPT_TEMPLATE_CONFIG = os.path.join(CONFIGS_PATH, 'model_configs/prompt_template_config.yaml')
#BASE_PROMPT_MODEL = os.path.join(MODELS_PATH, 'Llama-3.1-8B/original/')
#BASE_PROMPT_MODEL = os.path.join(MODELS_PATH, 'microsoft-Phi-3.5-mini-instruct/')

# main model related data paths
MODEL_OUTPUTS = os.path.join(DATA_PATH, 'model_outputs')
MODEL_TEST_OUTPUTS = os.path.join(DATA_PATH, 'model_test_outputs')
MAIN_MODEL_INPUT = os.path.join(MODEL_INPUT_PATH, 'main_model_input.json')
MAIN_MODEL_CONFIGS = os.path.join(MODEL_CONFIGS, 'main_model_config.yaml')
MAIN_MODEL = os.path.join(MODELS_PATH, 'microsoft-Phi-3.5-mini-instruct/')   
MAIN_MODEL_ID = "microsoft/Phi-3.5-mini-instruct"