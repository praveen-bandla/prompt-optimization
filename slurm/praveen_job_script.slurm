#!/bin/bash
#SBATCH --job-name=praveen-test-job       # Job name
#SBATCH --output=job_output.txt           # Standard output file
#SBATCH --error=job_error.txt            # Standard error file
#SBATCH --time=00:10:00                  # Time limit (10 minutes)
#SBATCH --partition=short                # Partition name (adjust as needed)
#SBATCH --ntasks=1                       # Number of tasks (processes)
#SBATCH --cpus-per-task=1                # Number of CPUs per task
#SBATCH --mem=1G                         # Memory per node (adjust as needed)

# Load necessary modules (if applicable)
# module load python/3.x

# Run the Python script using the default environment
python scripts/testing_hpc.py
