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
#MODEL_INPUTS = os.path.join(ROOT_PATH, 'model_input')

# Validation model related files
VALIDATION_SCORES = os.path.join(DATA_PATH, 'validation_scores')
VALIDATOR_MODEL_CONFIGS = os.path.join(CONFIGS_PATH, 'validator_model_config.yaml')

# BP and BPV related files
SQL_DB = os.path.join(DATA_PATH, 'base_prompts', 'base_prompts.db')
PROMPT_VARIATIONS = os.path.join(DATA_PATH, 'prompt_variations')
PROMPT_VARIATION_MODEL = os.path.join(MODELS_PATH, 'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B')
PROMPT_VARIATION_MODEL_INPUT = os.path.join(MODEL_INPUT_PATH, 'prompt_variation_input.json')
PROMPT_VARIATION_MODEL_CONFIG = os.path.join(CONFIGS_PATH, 'prompt_variation_config.yaml')
BASE_PROMPT_MODEL_INPUT = os.path.join(MODEL_INPUT_PATH, 'base_prompt_input.json')
BASE_PROMPT_MODEL_CONFIG = os.path.join(CONFIGS_PATH, 'model_configs/base_prompt_config.yaml')
#BASE_PROMPT_MODEL = os.path.join(MODELS_PATH, 'Llama-3.1-8B/original/')
BASE_PROMPT_MODEL = os.path.join(MODELS_PATH, 'microsoft-Phi-3.5-mini-instruct/')

# main model related data paths
MODEL_OUTPUTS = os.path.join(DATA_PATH, 'model_outputs')
MAIN_MODEL_INPUT = os.path.join(MODEL_INPUT_PATH, 'main_model_input.json')
MAIN_MODEL_CONFIGS = os.path.join(CONFIGS_PATH, 'main_model_config.yaml')
MAIN_MODEL = os.path.join(MODELS_PATH, 'microsoft-Phi-3.5-mini-instruct/')   