#!/bin/bash

#SBATCH --job-name=generate_prompt_variations_2
#SBATCH --output=slurm/output/generate_prompt_variations_2_%j.out
#SBATCH --error=slurm/output/generate_prompt_variations_2_%j.err
#SBATCH --partition=rtx8000
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=04:00:00
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
  /bin/bash -c "
    source /ext3/miniconda3/etc/profile.d/conda.sh
    conda activate venv-bella
    pip install -r /scratch/ic2664/prompt-optimization/requirements.txt
    cd /scratch/ic2664/prompt-optimization/
    export PYTHONPATH=\$PYTHONPATH:/scratch/ic2664/prompt-optimization/
    export HF_HOME=/scratch/ic2664/huggingface_cache
    echo 'Start time: ' \$(date)
    echo 'Running prompt variations for bp_idx [100, 101, ..., 199]'
    python .src/scripts/generate_prompt_variations_new.py '[100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159,160,161,162,163,164,165,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191,192,193,194,195,196,197,198,199]'
    echo 'End time: ' \$(date)
"
