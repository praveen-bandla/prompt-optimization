'''
This document contains the paths to the root directories of the project.
'''

import os


ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT_PATH, 'data')
MODELS_PATH = os.path.join(ROOT_PATH, 'models')
CONFIGS_PATH = os.path.join(ROOT_PATH, 'configs')
MODEL_INPUT_PATH = os.path.join(ROOT_PATH, 'model_input')

SQL_DB = os.path.join(DATA_PATH, 'base_prompts', 'base_prompts.db')
PROMPT_VARIATIONS = os.path.join(DATA_PATH, 'prompt_variations')