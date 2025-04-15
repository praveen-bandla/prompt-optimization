#!/bin/bash

#SBATCH --job-name=generate_base_prompts
#SBATCH --output=output/generate_base_prompts.out
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=1:00:00
#SBATCH --mem=32GB
#SBATCH --mail-type=END
#SBATCH --mail-user=dj2565@nyu.edu

# Purge any loaded modules
module purge
module load singularity

# Execute inside Singularity with overlay, GPU, and Python 3.10.6 environment
singularity exec --nv \
  --bind /scratch \
  --overlay /scratch/dj2565/overlay-25GB-500K.ext3:rw \
  /scratch/dj2565/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
  /bin/bash -c "
    source /ext3/miniconda3/etc/profile.d/conda.sh
    conda activate venv-darren
    cd /scratch/dj2565/prompt-optimization/
    python ./src/scripts/generate_base_prompts_new.py
"