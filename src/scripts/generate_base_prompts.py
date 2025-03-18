"""
Generate Base Prompts

This script generates a set of base prompts by calling a model to produce base prompts based on
the provided input. It runs inference and generates 'n' base prompts for further use.

No input: None, base prompts generated as a batch

Load as dependencies:
- `base_prompt_input.txt`: A text file containing the base instruction for generating prompts.
- `base_prompt_config.yaml`: A YAML file with configuration settings for prompt generation.
- data_handler.py base prompt class
- prompt.py base prompt class

Outputs: None

Overview:
Script pulls number of base prompts from configs, generates prompts, randomly shuffles them, and writes them to a SQLite database. 
  | bp_idx | base_prompt |
  |--------|-------------|
  | 1 | "Generated base prompt 1" |
  | 2 | "Generated base prompt 2" |
  | ...    | ...         |

Shuffling implementation:
- DON'T run through list of strings and randomly shuffle and write to SQL
- Shuffle indices 0 to n-1, then write:
for i in [shuffled_indices]:
    write to SQL strs[i]

- Opens and writes the generated prompts to `base_prompts.sqlite` in SQLite/database format for efficient storage. 

Usage:
- Ensure that both input files (`base_prompt_input.txt` and `base_prompt_config.yaml`) are in the same
  directory as the script before running it.
- The generated prompts will be saved to a SQLite file named `base_prompts.sqlite`.
- Have slurm config file ready to run the script on HPC
- Add all libraries I'm importing into requirements.txt

Dependencies:
- [List of libraries or modules, e.g. 'pandas', 'transformers', 'torch', etc.]
"""