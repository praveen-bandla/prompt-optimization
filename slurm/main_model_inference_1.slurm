#!/bin/bash

#SBATCH --job-name=main_model_inference_1
#SBATCH --output=slurm/output/main_model_inference_1_%j.out
#SBATCH --error=slurm/output/main_model_inference_1_%j.err
#SBATCH --partition=rtx8000
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --mem=96GB
#SBATCH --mail-type=END
#SBATCH --mail-user=dj2565@nyu.edu

# Clean environment
module purge
module load singularity/3.7.4

# Run inside Singularity container with overlay and Conda
singularity exec --nv \
  --bind /scratch \
  --overlay /scratch/dj2565/overlay-25GB-500K.ext3:ro \
  /scratch/dj2565/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
  /bin/bash -c "
    source /ext3/miniconda3/etc/profile.d/conda.sh
    conda activate venv-darren
    cd /scratch/dj2565/prompt-optimization/
    export PYTHONPATH=\$PYTHONPATH:/scratch/dj2565/prompt-optimization/
    export HF_HOME=/scratch/dj2565/huggingface_cache
    echo 'Start time: ' \$(date)
    echo 'Running main model inference for bp_idx [0, 1, ..., 99]'
    python src/scripts/main_model_inference.py 0 100
    echo 'End time: ' \$(date)
"
