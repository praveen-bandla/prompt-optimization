#!/bin/bash

#SBATCH --job-name=generate_prompt_variations_4
#SBATCH --output=slurm/output/generate_prompt_variations_4_%j.out
#SBATCH --error=slurm/output/generate_prompt_variations_4_%j.err
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
  /scratch/dj2565/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
  /bin/bash -c "
    source /ext3/miniconda3/etc/profile.d/conda.sh
    conda activate venv-bella
    pip install -r /scratch/ic2664/prompt-optimization/requirements.txt
    cd /scratch/ic2664/prompt-optimization/
    export PYTHONPATH=\$PYTHONPATH:/scratch/ic2664/prompt-optimization/
    export HF_HOME=/scratch/ic2664/huggingface_cache
    echo 'Start time: ' \$(date)
    echo 'Running prompt variations for bp_idx [300, 301, ..., 399]'
    python .src/scripts/generate_prompt_variations_new.py '[300,301,302,303,304,305,306,307,308,309,310,311,312,313,314,315,316,317,318,319,320,321,322,323,324,325,326,327,328,329,330,331,332,333,334,335,336,337,338,339,340,341,342,343,344,345,346,347,348,349,350,351,352,353,354,355,356,357,358,359,360,361,362,363,364,365,366,367,368,369,370,371,372,373,374,375,376,377,378,379,380,381,382,383,384,385,386,387,388,389,390,391,392,393,394,395,396,397,398,399]'
    echo 'End time: ' \$(date)
"
