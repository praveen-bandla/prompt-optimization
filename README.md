# [Paper Title]: Prompt Optimization for LLMs

## Project Overview

This project aims to optimize prompts for large language models (LLMs) through a model-based approach, involving generating synthetic data, training a prompt model, and testing it against the baseline. The focus is on automating prompt optimization in a way that is model-agnostic and scalable across different LLMs and tasks, without the reliance on any datasets. The approach integrates contrastive learning and continual learning techniques for enhanced prompt quality.

<br>

## Project Directory Structure

The project is organized into several directories and files to help maintain modularity and clarity. Below is an overview of each directory's purpose, along with the contents of each directory.

``` NB: A lot of this is due to change. Think of it as a working directory``` 

```md
/synthetic_prompt_optimization
│── configs/
│   ├── base_prompt_config.json
│   ├── prompt_variation_config.json
│   ├── main_model_config.json
│   ├── validator_model_config.json
│   ├── prompt_optimization_config.json
│   ├── fine_tuning_config.json
│── data/
│   ├── raw/
│   │   ├── base_prompts.parquet
│   │   ├── prompt_variations.parquet
│   │   ├── model_outputs.parquet
│   │   ├── validator_scores.parquet
│   ├── processed/
│   │   ├── train.parquet
│   │   ├── test.parquet
│   │   ├── validation.parquet
│   ├── database/
│   │   ├── main_model_outputs.db  # SQLite database (if needed)
│── models/
│   ├── main_model/
│   ├── validator_models/
│   ├── prompt_optimization_model/
│── scripts/
│   ├── 01_generate_base_prompts.py
│   ├── 02_generate_prompt_variations.py
│   ├── 03_main_model_inference.py
│   ├── 04_validator_model_inference.py
│   ├── 05_train_prompt_optimization.py
│   ├── 06_fine_tune_main_model.py
│   ├── 07_test_framework.py
│── utils/
│   ├── model_loader.py
│   ├── data_handler.py
│   ├── metrics.py
│   ├── logging_utils.py
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