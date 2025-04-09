#!/bin/bash

#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00  # Max run time
#SBATCH --mem=2GB  # Adjust memory as needed
#SBATCH --chdir=

#SBATCH --job-name=prose-test-bp-generation
#SBATCH --output=output.out

module purge
module load python/3.10.4

source becca-venv/bin/activate
python prompt-optimization/src/scripts/generate_base_prompts.py
