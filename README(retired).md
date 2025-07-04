# [Paper Title]: Prompt Optimization for LLMs

## Project Overview

This project aims to optimize prompts for large language models (LLMs) through a model-based approach, involving generating synthetic data, training a prompt model, and testing it against the baseline. The focus is on automating prompt optimization in a way that is model-agnostic and scalable across different LLMs and tasks, without the reliance on any datasets. The approach integrates contrastive learning and continual learning techniques for enhanced prompt quality.


<br>

## Project Directory Structure

The project is organized into several directories and files to help maintain modularity and clarity. Below is an overview of each directory's purpose, along with the contents of each directory.

``` NB: A lot of this is due to change. Think of it as a working directory``` 

### `config/`
This directory contains configuration files used to control various parameters in the project, including environment setup, model settings, and SLURM configurations for job submission.

- **`config.yaml`**: Stores project-specific configurations (e.g., model hyperparameters, paths). 
    - **This file would contain specifics of which models we are using for each task. Our code would modular in a way that works for any model we input**
    - the directories of the subfolders (./data, ./ouputs, etc.) and specific files. That way, our code works without needing the exact same repository
    - contains our parameters we use for each model/task
- **`settings.yaml`**: Contains application-wide settings for the project, like training options or input/output directories.
    - things like gpu specifications/environment?
- **`slurm_config.yaml`**: Defines parameters for SLURM jobs, including job type, resource allocation, etc.
    - apparently needed for running our jobs, not too sure how it works exactly but need to read more

### `data/`
This directory holds all data files required for the project, such as input data, synthetic data, and results from previous runs.

- **`synthetic_data.parquet`**: Holds synthetic data generated during the initial phase of the project, used for training.
    - never used parquet files. Apparently like JSON, less readable, more efficient
- **`training_data.pkl`**: Contains the processed data that will be used to train the prompt model.
    - we may not need this if we standardize our data formats well, but leaving this here for now
- **`comparison_results.parquet`**: Stores results of comparisons between different models or approaches.
    - scores for each output. I don't think we need to store our outputs in our final synthetic data file, so we can store it here instead. 

### `scripts/`
This directory contains all the Python scripts for running various tasks in the project, including data generation, model training, evaluation, and inference.

- **`/synthetic_data`**: Contains scripts for all sub-tasks involved with synthetic data generation
- **`generate_synthetic_data.py`**: Generates synthetic data to be used in training or testing. Executes all tasks in the `./scripts/synthetic_data/` subfolder by running all scripts.
- **`run_inference.py`**: Runs inference on main model for a given input prompt. This could be baseline prompt or refined prompt (post prompt-model processing)
- **`train_prompt_model.py`**: Contains code for training the prompt model.
- **`evaluate_performance.py`**: Evaluates the performance of the model on the test data.
- **`submit_slurm_job.py`**: Submits jobs to a SLURM cluster for distributed computing.
    - again, need to learn how this works

### `tests/`
This directory contains test files for various components of the project, ensuring code correctness and model performance. We can get rid of many of these files if we feel it is overkill. It is very good practice though and prevents us from running into severe issues.

- **`test_synthetic_data.py`**: Unit tests for synthetic data generation processes. 
- **`test_prompt_variation.py`**: Tests for the functionality of prompt variations.
- **`test_inference.py`**: Tests for inference-related functionality and correctness.
- **`test_validation.py`**: Tests to ensure the validation process runs as expected.
- **`test_training.py`**: Contains tests for the training processes.
- **`test_end_to_end.py`**: End-to-end tests to check if the complete pipeline works as expected.

### `notebooks/`
Contains Jupyter notebooks for experimentation, visualization, and analysis. These notebooks may be used for exploratory work or to document the results of specific experiments. We can start with some work here before moving to scripts

### `logs/`
This directory is used to store logs generated by various scripts, which can help in debugging or analyzing the results of model training and inference. May not need but apparently good practice

### `checkpoints/`
Holds model checkpoints saved during training, allowing you to resume training from the latest checkpoint or evaluate models at different training stages. Good practice and helps us split up our training process and evaluate step by step

### `tokenizer/`
Contains tokenization logic and related files, which are used to preprocess text data for the model. Apparently these are always needed for fine-tuning language models, but since we will be using an open source as base, we just import/duplicate the existing one we see. Never did it before personally. 

### Root Files

- **`README.md`**: This file! It provides an overview of the project, instructions, and additional resources. *How meta*
- **`.gitignore`**: A file that specifies which files and directories should be ignored by Git, ensuring sensitive or unnecessary files (such as the environment or large data files) are not tracked. Ignore the prompt-optimization-env/ line - I tried to make an environment but that didn't work and I got annoyed so I deleted. We can add later
- **`env/` (optional)**: This is now excluded from version control. We may still create an environment locally as needed for Python dependencies to ensure no dependency issues.
- **`requirements.txt` (optional)**: This file would contain all the packages we download. It is really good practice but I lost my brain trying to fix a bug related to the enviornment. If someone knows anything lmk

---

<!-- ## Conventions for Team Members

1. **Branching**: Each feature or bugfix should be done in a separate branch. For example:
   - Create a branch for new features: `git checkout -b feature/branch-name`
   - Create a branch for bugfixes: `git checkout -b bugfix/issue-name`

2. **Commit Messages**: Follow a clear and consistent format when committing. Example:
   - `feat: Add new model evaluation script`
   - `fix: Correct bug in synthetic data generation`
   - `chore: Update README with new folder structure`
   - `test: Add unit tests for inference`

3. **Environment Setup**: Always create a new environment using `requirements.txt` if needed. To set up:
   - `python3 -m venv env` (if you decide to use a virtual environment again)
   - `pip install -r requirements.txt` to install dependencies.

4. **Data Handling**: Never commit large data files (e.g., `.pkl`, `.parquet`). Always add them to `.gitignore` to avoid accidental commits.

5. **Testing**: Ensure that all new code is covered by appropriate tests. Run tests using `pytest` or your preferred testing framework before pushing. -->

## Git Conventions for Collaboration

### Branching

When working on the project, it's important that everyone creates their own **branch** before making any changes. This helps keep the `main` branch stable and avoids conflicts when merging. Here's how to create and work with branches:

#### 1. **Creating a New Branch**

(One-time work)
To create a new branch, follow these steps:

1. Make sure you’re on the `main` branch:
    ```bash
    git checkout main
    ```
2. Pull the latest changes from the remote repository to make sure you're up to date:
    ```bash
    git pull origin main
    ```
3. Create a new branch in this format:
    ```bash
    git checkout -b {your_name}-branch
    ```
    If for whatever reason you need a subranch for a specific feature or something, you can use:

    ```bash
    git checkout -b {your_name}-branch-{feature}
    ```


#### 2. **Accessing a Branch**

To switch to a different branch (either after creating one or accessing an existing one):

1. List all available branches:
    ```bash
    git branch
    ```
    - This will show you all local branches in your project. The current branch will be marked with an asterisk (`*`).

2. To switch to an existing branch, use:
    ```bash
    git checkout <branch-name>
    ```
    - Replace `<branch-name>` with the name of the branch you want to work on. For example, `feature-login`.

3. If you're unsure about which branch you're on, you can check the active branch with:
    ```bash
    git branch
    ```
    - The active branch will have a `*` next to its name.

#### 3. **Making Changes**

1. Once you are in your branch, you can make changes, add new files, or modify existing files as needed.
2. Regularly commit small, logical changes to keep your work organized and easy to understand.

    **Example**:
    ```bash
    git add <file-name>      # Adds a specific file
    git add .                # Adds all changes
    git commit -m "Description of changes"  # Commits changes with a meaningful message
    ```

#### 4. **Pushing Changes**

When you're ready to share your changes, push your branch to the remote repository:

```bash
git push --set-upstream origin <branch-name>
```

This basically pushes the updated changes to your branch. Once thats done, we can approve the changes via github directly.

<br>

### Data Handling

When working with large files, it's important to manage them carefully to avoid pushing excessive amounts of data at once. Here are some best practices to follow:

1. **Avoid committing too many changes at once**:
    - It's better to commit changes incrementally rather than pushing a large number of changes in one go. This will make it easier to track progress and reduce the chances of merge conflicts.

2. **Git's file size cap**:
    - GitHub has a file size limit of 100 MB for regular files. If a file exceeds this limit, it won't be able to be pushed to the repository.
    - If you're working with large files that exceed this limit, consider using [Git Large File Storage (LFS)](https://git-lfs.github.com/) or managing them in another way.

3. **Handling files larger than 100 MB**:
    - If a file is much larger than 100 MB, it should be added to `.gitignore` to prevent accidental commits. This ensures we are still able to work with it but its not public on our git.
    - For sharing large files, we can upload them to a cloud storage service like Google Drive or Dropbox and share the link with your team.
    - Be sure to include a reference to the file in the project, such as the link to the shared file on Google Drive or Dropbox.





