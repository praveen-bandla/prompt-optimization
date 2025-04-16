#!/bin/bash

#SBATCH --job-name=generate_base_prompts_rtx
#SBATCH --output=output/generate_base_prompts.out
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=1:00:00
#SBATCH --mem=32GB
#SBATCH --mail-type=END
#SBATCH --mail-user=ic2664@nyu.edu

# Purge any loaded modules
module purge
module load singularity

# Execute inside Singularity with overlay, GPU, and Python 3.10.6 environment
singularity exec --nv \
  --bind /scratch \
  --overlay /scratch/ic2664/overlay-25GB-500K.ext3:rw \
  /scratch/ic2664/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
  /bin/bash -c "
    source /ext3/miniconda3/etc/profile.d/conda.sh
    conda activate venv-bella
    cd /scratch/ic2664/prompt-optimization/
    python ./src/scripts/generate_base_prompts_new.py
"
