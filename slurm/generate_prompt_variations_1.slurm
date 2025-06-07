#!/bin/bash

#SBATCH --job-name=generate_prompt_variations_1
#SBATCH --output=slurm/output/generate_prompt_variations_1_%j.out
#SBATCH --error=slurm/output/generate_prompt_variations_1_%j.err
#SBATCH --partition=rtx8000
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=04:00:00
#SBATCH --mem=64GB
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
    echo 'Running prompt variations for bp_idx [0, 1, ..., 99]'
    python src/scripts/generate_prompt_variations_new.py '[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99]'
    echo 'End time: ' \$(date)
"
