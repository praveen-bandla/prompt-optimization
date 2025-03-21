'''
This document contains the paths to the root directories of the project.
'''

import os


ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT_PATH, 'data')
MODELS_PATH = os.path.join(ROOT_PATH, 'models')
CONFIGS_PATH = os.path.join(ROOT_PATH, 'configs')
MODEL_INPUT_PATH = os.path.join(ROOT_PATH, 'model_input')
#MODEL_INPUTS = os.path.join(ROOT_PATH, 'model_input')

VALIDATION_SCORES = os.path.join(DATA_PATH, 'validation_scores')
VALIDATOR_MODEL_CONFIGS = os.path.join(CONFIGS_PATH, 'validator_model_config.yaml')


SQL_DB = os.path.join(DATA_PATH, 'base_prompts', 'base_prompts.db')
PROMPT_VARIATIONS = os.path.join(DATA_PATH, 'prompt_variations')

MODEL_OUTPUTS = os.path.join(DATA_PATH, 'model_outputs')
MAIN_MODEL_INPUT = os.path.join(MODEL_INPUTS, 'main_model_input.json')
MAIN_MODEL = os.path.join(MODELS_PATH, 'microsoft-Phi-3.5-mini-instruct/')
MAIN_MODEL_CONFIGS = os.path.join(CONFIGS_PATH, 'main_model_config.yaml')
