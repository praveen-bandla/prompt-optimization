#!/bin/bash

#SBATCH --job-name=generate_base_prompts
#SBATCH --output=slurm/output/generate_base_prompts_%j.out
#SBATCH --error=slurm/output/generate_base_prompts_%j.err
#SBATCH --partition=rtx8000
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=02:00:00
#SBATCH --mem=64GB
#SBATCH --mail-type=END
#SBATCH --mail-user=dj2565@nyu.edu

# Clean environment
module purge
module load singularity/3.7.4

# Run inside Singularity container with overlay and Conda
singularity exec --nv \
  --bind /scratch \
  --overlay /scratch/dj2565/overlay-25GB-500K.ext3:rw \
  /scratch/dj2565/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
  /bin/bash -c "
    source /ext3/miniconda3/etc/profile.d/conda.sh
    conda activate venv-darren
    cd /scratch/dj2565/prompt-optimization/
    export PYTHONPATH=$PYTHONPATH:/scratch/dj2565/prompt-optimization/
    echo 'Start time: ' \$(date)
    python ./src/scripts/generate_base_prompts_new.py
    echo 'End time: ' \$(date)
"
