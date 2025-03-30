# Downloading phi 3-5

from transformers import AutoModelForCausalLM, AutoTokenizer

# Define the local directory to store the model

# change this as needed
net_id = "pb3060"
local_model_dir = f"/scratch/{net_id}/prompt-optimization/models/microsoft-Phi-3.5-mini-instruct/"

# Download the model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3.5-mini-instruct", 
    cache_dir=local_model_dir,
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(
    "microsoft/Phi-3.5-mini-instruct", 
    cache_dir=local_model_dir,
    trust_remote_code=True
)


print("Model and tokenizer downloaded and saved locally.")