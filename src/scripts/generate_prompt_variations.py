"""
Generate Prompt Variations

This script generates a set of prompt variations by calling the prompt variation model to run inference/generate 'n' prompt variations. 

Inputs:
- `prompt_variation_input.txt`: A text file containing the base instruction for generating prompt variations.
- `prompt_variation_config.yaml`: A YAML file with configuration settings for prompt variation generation.

Outputs:
- Opens and writes the generated prompts to `prompt_variations.parquet` in Parquet format for efficient storage. 
  The output includes `bpv_idx`, which is a tuple in the format `(bp_idx, index)`, where `index` is an integer 
  between 1 and `n`, representing the nth variation. Each entry will also include a string `prompt_variation: str`.

Usage:
- Ensure that both input files (`base_prompt_input.txt` and `base_prompt_config.yaml`) are in the same
  directory as the script before running it.
- The generated prompts will be saved to a Parquet file named `base_prompts.parquet`.

Dependencies:
- [List of libraries or modules, e.g. 'pandas', 'transformers', 'torch', etc.]
- Needs the index of the base prompt
"""