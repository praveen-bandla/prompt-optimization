#!/bin/bash

#SBATCH --error=output/generate_base_prompts.err
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
	echo 'Step 1: Inside Singularity'
	source /ext3/miniconda3/etc/profile.d/conda.sh || echo 'Failed to source conda.sh'
	which conda || echo 'conda not found'
	conda activate venv-bella || echo 'Failed to activate conda env'
	which python || echo 'python not found'
	cd /scratch/ic2664/prompt-optimization/ || echo 'Failed to cd'
	python ./src/scripts/generate_base_prompts_new.py || echo 'Python script crashed'
"
