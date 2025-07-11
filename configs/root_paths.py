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
FILTERED_SQL_DB = os.path.join(DATA_PATH, 'base_prompts', 'filtered_base_prompts.db')
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
MAIN_MODEL_INPUT = os.path.join(MODEL_INPUT_PATH, 'main_model_input.json')
MAIN_MODEL_CONFIGS = os.path.join(MODEL_CONFIGS, 'main_model_config.yaml')
MAIN_MODEL = os.path.join(MODELS_PATH, 'microsoft-Phi-3.5-mini-instruct/')   
MAIN_MODEL_ID = "microsoft/Phi-3.5-mini-instruct"

# prompt optimizer related paths
REGRESSION_HEAD_BASE_MODEL_ID = 'meta-llama/Llama-3.2-1B'
REGRESSION_HEAD_BASE_PATH = os.path.join(MODELS_PATH, 'prompt_opt_base')
PROMPT_GEN_BASE_MODEL_ID = 'meta-llama/Llama-3.2-1B-Instruct'
PROMPT_GEN_BASE_PATH = os.path.join(MODELS_PATH, 'prompt_gen_base')
LORA_PROMPT_GEN_PATH = os.path.join(MODELS_PATH, 'lora_prompt_gen')
LORA_REGRESSION_HEAD_PATH = os.path.join(MODELS_PATH, 'lora_regression_head')
REGRESSION_HEAD_CONFIG_PATH = os.path.join(MODEL_CONFIGS, 'regression_head.yaml')
BEST_LORA_REGRESSION_HEAD_PATH = os.path.join(MODELS_PATH, 'lora_regression_head_best_trial')
BEST_LORA_PROMPT_GEN_PATH = os.path.join(MODELS_PATH, 'lora_prompt_gen_best_trial')
PROMPT_GENERATOR_MODEL_INPUT = os.path.join(MODEL_INPUT_PATH, 'prompt_optimizer_input.json')
SAVE_PATH = f"{LORA_REGRESSION_HEAD_PATH}_best_trial"
REGRESSION_HEAD_PATH = os.path.join(SAVE_PATH, 'regression_head.pth')

VALIDATION_TEST_SCORES = os.path.join(DATA_PATH, 'validation_test_scores')

ANALYSIS_PATH = os.path.join(ROOT_PATH, 'analysis_data')