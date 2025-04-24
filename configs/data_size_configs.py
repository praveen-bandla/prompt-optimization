'''
This script contains all information related to the data size configurations. I have added some example things but this is not complete. This is just a starting point. Add to it as needed.
'''

NUM_PROMPT_VARIATIONS = 50
NUM_BASE_PROMPTS = 2000
NUM_BATCHES = 50
BATCH_SIZE = 40
MM_OUTPUT_BATCH_SIZE = 10


# PARTITION_SIZE_MODEL_OUTPUT_PARQUET = 0
NUM_RUBRIC_SECTIONS = 5
SECTION_WEIGHTS = {
    'section_1': 1.0,
    'section_2': 1.0,
    'section_3': 0.5,
    'section_4': 2.0,
    'section_5': 0.5
}
NUM_VALIDATOR_MODELS = 2

# Map model aliases to model names
# commenting out for now
# MODEL_ALIASES = {
#     'validator_model_1': 'falcon_mamba',
#     'validator_model_2': 'opt',
#     'validator_model_3': 'mistral_instruct',
# }


VAL_MODEL_DICT = {
    0:{
        'model_name': 'falcon_mamba',
        # 'huggingface_model_id': 'tiiuae/falcon-mamba-7b',
        'huggingface_model_id': 'tiiuae/Falcon3-3B-Instruct',
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
        'model_name': 'gemma',
        'huggingface_model_id': 'google/gemma-2-9b-it',
        'prompt_structure':[
            {
                "role": "user",
                "content": ""
            } 
        ]
    }
#     },
#     2:{
#         'model_name': 'qwen',
#         'huggingface_model_id': 'Qwen/Qwen2.5-7B-Instruct',
#         'prompt_structure':[
#             {
#                 "role": "system",
#                 "content": ""
#             },
#             {
#                 "role": "user",
#                 "content": ""
#             }
#         ]
#     }
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