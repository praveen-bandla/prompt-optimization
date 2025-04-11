'''
This script contains all information related to the data size configurations. I have added some example things but this is not complete. This is just a starting point. Add to it as needed.
'''

NUM_PROMPT_VARIATIONS = 10
NUM_BASE_PROMPTS = 10
# PARTITION_SIZE_MODEL_OUTPUT_PARQUET = 0
NUM_RUBRIC_SECTIONS = 5
SECTION_WEIGHTS = {
    'section_1': 0.2,
    'section_2': 0.2,
    'section_3': 0.2,
    'section_4': 0.2,
    'section_5': 0.2
}
NUM_VALIDATOR_MODELS = 3

# Map model aliases to model names
# commenting out for now
MODEL_ALIASES = {
    'validator_model_1': 'falcon_mamba',
    'validator_model_2': 'opt',
    'validator_model_3': 'mistral_instruct',
}


VAL_MODEL_DICT = {
    0:{
        'model_name': 'falcon_mamba',
        'huggingface_model_id': 'tiiuae/falcon-mamba-7b',
        'prompt_structure':[
            {
                "role": "system",
                "content": ""
            },
            {
                "role": "user",
                "content": ""
            }
        ]
            
    },  

    1:{
        'model_name': 'opt',
        'huggingface_model_id': 'facebook/opt-6.7b',
        'prompt_structure':[
            {
                "role": "system",
                "content": ""
            },
            {
                "role": "user",
                "content": ""
            } 
        ]
    },
    2:{
        'model_name': 'mistral_instruct',
        'huggingface_model_id': 'mistralai/Mistral-7B-Instruct-v0.2',
        'prompt_structure':[
            {
                "role": "user",
                "content": ""
            }
        ]
    }
}

# MODEL_ALIASES = {
#     'validator_model_1': {
#         'model_name': 'falcon_mamba',
#         'model_id': 'tiiuae/falcon-mamba-7b',
#         'prompt_structure':{
#             'role': 'user',
#             'content': {system_role}{content}
#         }
#     },
#     }'falcon_mamba',
#     'validator_model_2': 'opt',
#     'validator_model_3': 'mistral_instruct',
# }