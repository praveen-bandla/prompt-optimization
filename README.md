# PROSE: Prompt Optimization for LLMs

## Project Overview

This project aims to optimize prompts for large language models (LLMs) through a model-based approach, involving generating synthetic data, training a prompt model, and testing it against the baseline. The focus is on automating prompt optimization in a way that is model-agnostic and scalable across different LLMs and tasks, without the reliance on any pre-existing datasets. The approach integrates contrastive learning and continual learning techniques for enhanced prompt quality.

<br>

## Project Directory Structure

The project is organized into several directories and files to help maintain modularity and clarity. Below is an overview of each file, along with the contents of each directory.

``` NB: A lot of this is due to change. Think of it as a working directory``` 

```md
/prompt-optimization
│── model_input/
│   ├── base_prompt_input.txt
│   ├── prompt_variation_input.txt
│   ├── main_model_input.txt
│   ├── rubric.txt
│   ├── val_model_input.txt
│   ├── prompt_optimization_model_input.txt
│   ├── finetuning_input.txt
│── configs/
│   ├── root_paths.py
│   ├── data_size_configs.py
│   ├── settings.yaml
│   ├── model_configs/
│       ├── base_prompt_config.yaml
│       ├── prompt_variation_config.yaml
│       ├── main_model_config.yaml
│       ├── validator_model_config.yaml
│       ├── prompt_optimization_config.yaml
│       ├── fine_tuning_config.yaml
│── slurm/
│   ├── sbatch_files/
│   ├── stdouts/
│── data/
│   ├── base_prompts/
│   │   ├── base_prompts.db
│   ├── prompt_variations/
│   │   ├── {base_prompt_id}_pv.parquet
│   ├── validator_scores/
│   │   ├── {base_prompt_id}_vs.parquet
│   ├── model_outputs/
│   │   ├── {base_prompt_id}_mo.parquet
│── models/
│   ├── base_prompt_model/
│   ├── prompt_variation_model/
│   ├── main_model/
│   ├── validator_models/
│   │   ├── val_model_1/
│   │   ├── val_model_2/
│   │   ├── val_model_3/
│   ├── prompt_optimization_model/
│── src/
│   ├── utils/
│   │   ├── model_loader.py
│   │   ├── data_handler.py
│   │   ├── metrics.py
│   │   ├── logging_utils.py
│   │   ├── prompt.py
│   │   ├── inference.py
│   ├── scripts/
│   │   ├── generate_prompt_variations.py
│   │   ├── generate_base_prompts.py
│   │   ├── main_model_inference.py
│   │   ├── validator_model_inference.py
│   │   ├── train_prompt_optimization.py
│   │   ├── fine_tune_main_model.py
│   │   ├── test_framework.py
│   │   ├── download_models.py
│── results/
│   ├── logs/
│   ├── evaluations/
│   ├── test_results.parquet
│── notebooks/
│   ├── exploratory_data_analysis.ipynb
│   ├── model_debugging.ipynb
│── requirements.txt
│── README.md
│── .gitignore
│── setup.py
```

Here is a guide on some of the key files/folders.

1. `model_input`: a folder that stores instructions (as text files) for each model that is being inferenced.

2. `configs`: a folder that stores the configurations of each model that is to be inferenced or trained. This would include model name, hyperparameters, location, etc.

3. `slurm`: all slurm files used to run jobs. Exact file system tbd

4. `data`: detailed guide below

5. `models`: a folder used to store all model related files necessary for inference. This could include weights, checkpoints, etc. What files these are depends heavily on the model being used as well as the task (train or inference).

6. `src`: the `src/scripts` folder contains all the scripts used to execute the various subcomponents of our project. The `utils` subfolders contains helper functions, or any methods or variables that are to be used for running the scripts.

7. `results`: folder containing all results related information that we might need for future use.

8. `notebooks`: any jupyter notebooks we may want to use for our own personal exploration (recommended practice is to add them to `.gitignore` so they are not on the repository)


### Data Structures

> I am leaving out explanations of why we chose what we did. I will only outline the specific file system.

We use a singular index that every record of a data to a unique combination of base prompt and prompt variation. Henceforth, we call this **`bpv_idx`**. `bpv_idx` is a tuple in the format: 

`(<base_prompt_idx>, <prompt_variation_idx>)`

The base prompt is also provided a bpv_idx of type
`(<base_prompt_idx>, -1)`

This index will be used for all files below.

1.  `data/base_prompts/`: stores all information related to base prompts. Contains a singular `SQLite` database with 2 columns: `bpv_idx`, and `base_prompt_string`
    - NB: we establish that only one group member can write to this file


2. `data/prompt_variations/`: stores all information related to prompt variations. Contains $n$ number of `parquet` files corresponding to $n$ base prompts. In other words, one `parquet` file here represents all prompt variations of a given base prompt. Each `parquet` file would contain two columns: `bpv_idx` and `prompt_variation_string`.


3. `data/validator_scores/`: stores all information related to validator scores. Contains $n$ number of `parquet` files corresponding to $n$ base prompts. In other words, one `parquet` file here represents scores for each prompt variations of a given base prompt. Each `parquet` file would contain 6 columns (number of rubric criteria + 1): `bpv_idx`, `criteria_1`,..., `criteria_5`. The values in each of `criteria_1`,..., `criteria_5` would be a tuple of `(<val_model_1_score>, <val_model_2_score>, <val_model_3_score>)`
    - NB: here, we generate all validator model inferences for a given `bpv_idx` before writing to this file.

4. `data/model_outputs`: stores all information related to model outputs. Contains $n$ number of `parquet` files corresponding to $n$ base prompts. In other words, one `parquet` file here represents all model outputs of a given prompt variation (this also includes the base prompt). Each `parquet` file would contain two columns: `bpv_idx` and `model_output_string`.