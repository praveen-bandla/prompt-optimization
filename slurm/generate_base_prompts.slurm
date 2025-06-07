#!/bin/bash


#SBATCH --job-name=generate_base_prompts
#SBATCH --output=slurm/output/generate_base_prompts_%j.out
#SBATCH --error=slurm/output/generate_base_prompts_%j.err
#SBATCH --partition=rtx8000

#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=04:00:00
#SBATCH --mem=64GB
#SBATCH --mail-type=END

## INSERT NETID EMAIL (xx@nyu.edu) HERE WHERE XXXX
#SBATCH --mail-user=XXXX



## WHERE XX -> netid (non-email)
## WHERE YY -> hpc venv (originally was venv-darren)
# Clean environment
module purge
module load singularity/3.7.4

# Run inside Singularity container with overlay and Conda
singularity exec --nv \
  --bind /scratch \
  --overlay /scratch/XX/overlay-25GB-500K.ext3:rw \
  /scratch/XX/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
  /bin/bash -c "
    source /ext3/miniconda3/etc/profile.d/conda.sh
    conda activate YY
    pip install -r /scratch/XX/prompt-optimization/requirements.txt
    cd /scratch/XX/prompt-optimization/
    export PYTHONPATH=$PYTHONPATH:/scratch/XX/prompt-optimization/
    echo 'Start time: ' \$(date)
    python src/scripts/generate_base_prompts_new.py
    echo 'End time: ' \$(date)
"
