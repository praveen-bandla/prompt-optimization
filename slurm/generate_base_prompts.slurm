#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=generate_base_prompts

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1

#SBATCH --time=1:00:00 
#SBATCH --mem=64GB 
#SBATCH --mail-type=END
#SBATCH --mail-user=ic2664@nyu.edu

module purge                        # Ensures a clean running environment
module load python/intel/3.8.6      # Loads the module for the software we are using

source ./venv_bella/bin/activate
python ./src/scripts/generate_base_prompts.py
