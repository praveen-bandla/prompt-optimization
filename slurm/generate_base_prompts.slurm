#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4 
#SBATCH --job-name=generate_base_prompts_a100
#SBATCH --output=output/generate_base_prompts_a100.out 

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

#SBATCH --time=0:30:00
#SBATCH --mem=32GB 

#SBATCH --mail-type=END
#SBATCH --mail-user=ic2664@nyu.edu

module purge                        # Ensures a clean running environment
module load python/intel/3.8.6      # Loads the module for the software we are using

source ./venv_bella/bin/activate
python ./src/scripts/generate_base_prompts_new.py
