#!/bin/bash

#SBATCH --job-name=generate_base_prompts
#SBATCH --output=slurm/output/generate_base_prompts_%j.out
#SBATCH --error=slurm/output/generate_base_prompts_%j.err
#SBATCH --partition=a100_2
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --mem=32GB
#SBATCH --mail-type=END
#SBATCH --mail-user=dj2565@nyu.edu

# Clean environment
module purge
module load singularity

# Run inside Singularity container with overlay and Conda
singularity exec --nv \
  --bind /scratch \
  --overlay /scratch/dj2565/overlay-25GB-500K.ext3:rw \
  /scratch/dj2565/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
  /bin/bash -c "
    source /ext3/miniconda3/etc/profile.d/conda.sh
    conda activate venv-darren
    cd /scratch/dj2565/prompt-optimization/
    echo 'Start time: ' \$(date)
    python ./src/scripts/generate_base_prompts_new.py
    echo 'End time: ' \$(date)
"
