#!/bin/bash

#SBATCH --job-name=validator_test_4
#SBATCH --output=slurm/output/validator_test_4_%j.out
#SBATCH --error=slurm/output/validator_test_4_%j.err
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
  --overlay /scratch/ic2664/overlay-25GB-500K.ext3:rw \
  /scratch/ic2664/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
  /bin/bash -c '
    cd /scratch/ic2664/prompt-optimization/
    source venv-bella/bin/activate
    export PYTHONPATH=$(pwd)
    echo "PYTHONPATH: $PYTHONPATH"
    export HF_HOME=/scratch/ic2664/huggingface_cache
    echo "Start time: $(date)"
    echo "Running validator score inference for bp_idx [800, 801, ..., 899]"
    python3 src/scripts/validator_model_inference_test.py 800 900
    echo "End time: $(date)"
'