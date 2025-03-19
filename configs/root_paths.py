'''
This document contains the paths to the root directories of the project.
'''

import os


ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT_PATH, 'data')
MODELS_PATH = os.path.join(ROOT_PATH, 'models')
CONFIGS_PATH = os.path.join(ROOT_PATH, 'configs')

SQL_DB = os.path.join(DATA_PATH, 'base_prompts', 'base_prompts.db')
PROMPT_VARIATIONS = os.path.join(DATA_PATH, 'prompt_variations')
MAIN_MODEL_CONFIGS = os.path.join(CONFIGS_PATH, 'main_model_config.yaml')
MAIN_MODEL = os.path.join(MODELS_PATH, 'microsoft-Phi-3.5-mini-instruct/')