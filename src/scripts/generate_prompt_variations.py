"""
Generate Prompt Variations

This script generates a set of prompt variations by calling the prompt variation model to run inference/generate 'n' prompt variations. 

Inputs:
- `prompt_variation_input.txt`: A text file containing the base instruction for generating prompt variations.
- `prompt_variation_config.yaml`: A YAML file with configuration settings for prompt variation generation.
- Section of prompt SQL DB from `generate_base_prompts.py`, bp_idx from 0 to n, n defined in config file

Outputs:
- Opens and writes the generated prompts to a series of Parquet files, one per base prompt.
- The output files will be named `prompt_variations_{n}.parquet`, where {n} is the number of the base prompt.
  The output includes `bpv_idx`, which is a tuple in the format `(bp_idx, index)`, where `index` is an integer 
  between 1 and `n`, representing the nth variation. Each entry will also include a string `prompt_variation: str`.
  Example format:
  | bpv_idx | prompt_variation |
  |---------|------------------|
  | (1, 1) | "Generated prompt variation 1" |
  | (1, 2) | "Generated prompt variation 2" |
  | ...     | ...              |

Overview:
- Inference prompt_variation_config.yaml and prompt_variation_input.txt to generate variations for a given base prompt and write to a .parquet file using functions from data_handler.py
- function to generate prompt variations takes one bp_idx number
  - Creates one PV-Parquet object (handles info from one specific prompt variation)
  - Creates 200 PV objects using strings from inferencing
- this only needs to do for one base prompt, we will call this script in a for loop for all base prompts we're looking at

Usage:
- Ensure that both input files (`base_prompt_input.txt` and `base_prompt_config.yaml`) are in the same
  directory as the script before running it.
- The generated prompts will be saved to a Parquet file named `prompt_variations_{n}.parquet`.

Dependencies:
- [List of libraries or modules, e.g. 'pandas', 'transformers', 'torch', etc.]
- Needs the index of the base prompt
"""