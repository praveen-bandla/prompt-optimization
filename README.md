# PROSE: Prompt Optimization for LLMs

## Project Overview

This project aims to optimize prompts for large language models (LLMs) through a model-based approach, involving generating synthetic data, training a prompt model, and testing it against the baseline. The focus is on automating prompt optimization in a way that is model-agnostic and scalable across different LLMs and tasks, without the reliance on any pre-existing datasets. The approach integrates contrastive learning and continual learning techniques for enhanced prompt quality.

<br>

## Project Directory Structure

The project is organized into several directories and files to help maintain modularity and clarity. Below is an overview of each file, along with the contents of each directory.

``` NB: A lot of this is due to change. Think of it as a working directory``` 

```md
/synthetic_prompt_optimization
│── model_input/
│   ├── base_prompt_input.txt
│   ├── prompt_variation_input.txt
│   ├── main_model_input.txt
│   ├── rubric.txt
│   ├── val_model_input.txt
│   ├── prompt_optimization_model_input.txt
│   ├── finetuning_input.txt
│── configs/
│   ├── base_prompt_config.yaml
│   ├── prompt_variation_config.yaml
│   ├── main_model_config.yaml
│   ├── validator_model_config.yaml
│   ├── prompt_optimization_config.yaml
│   ├── fine_tuning_config.yaml
│── slurm/
│   ├── sbatch_files/
│   ├── stdouts/
│── data/
│   ├── raw/
│   │   ├── base_prompts.parquet
│   │   ├── prompt_variations.parquet
│   │   ├── model_outputs.parquet
│   │   ├── validator_scores.parquet
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
│   ├──  utils/
│   │   ├── model_loader.py
│   │   ├── data_handler.py
│   │   ├── metrics.py
│   │   ├── logging_utils.py
│   │   ├── prompt.py
│   │   ├── inference.py
│   ├── scripts/
│   │   ├── generate_prompt_variations.py
│   │   ├── main_model_inference.py
│   │   ├── validator_model_inference.py
│   │   ├── train_prompt_optimization.py
│   │   ├── fine_tune_main_model.py
│   │   ├── test_framework.py
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

