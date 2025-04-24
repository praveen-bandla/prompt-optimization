#!/bin/bash

#SBATCH --job-name=validator_test_6
#SBATCH --output=slurm/output/validator_test_6_%j.out
#SBATCH --error=slurm/output/validator_test_6_%j.err
#SBATCH --partition=rtx8000
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=03:00:00
#SBATCH --mem=64GB
#SBATCH --mail-type=END
#SBATCH --mail-user=ic2664@nyu.edu

# Clean environment
module purge
module load singularity/3.7.4

# Run inside Singularity container with overlay and Conda
singularity exec --nv \
  --bind /scratch \
  --overlay /scratch/ic2664/overlay-25GB-500K.ext3:ro \
  /scratch/ic2664/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
  /bin/bash -c "
    source /ext3/miniconda3/etc/profile.d/conda.sh
    conda activate venv-bella
    cd /scratch/ic2664/prompt-optimization/
    export PYTHONPATH=\$PYTHONPATH:/scratch/ic2664/prompt-optimization/
    export HF_HOME=/scratch/ic2664/huggingface_cache
    echo 'Start time: ' \$(date)
    echo 'Running validator scoring for bp_idx [1000, 1001, ..., 1099]'
    python src/scripts/validator_model_inference.py 1000 1100
    echo 'End time: ' \$(date)
"