#!/bin/bash

#SBATCH --job-name=generate_prompt_variations_3
#SBATCH --output=slurm/output/generate_prompt_variations_3_%j.out
#SBATCH --error=slurm/output/generate_prompt_variations_3_%j.err
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
    echo 'Running prompt variations for bp_idx [200, 201, ..., 299]'
    python .src/scripts/generate_prompt_variations_new.py '[200,201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,216,217,218,219,220,221,222,223,224,225,226,227,228,229,230,231,232,233,234,235,236,237,238,239,240,241,242,243,244,245,246,247,248,249,250,251,252,253,254,255,256,257,258,259,260,261,262,263,264,265,266,267,268,269,270,271,272,273,274,275,276,277,278,279,280,281,282,283,284,285,286,287,288,289,290,291,292,293,294,295,296,297,298,299]'
    echo 'End time: ' \$(date)
"
