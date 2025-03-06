# Submitting HPC Jobs

## Batch vs Interactive Jobs
There are two kinds of jobs that we can submit: batch and interactive.
#### Interactive:
* Analogous to when you click something, computer performs the action, click something else, computer performs next action. 
* Working with HPCs is less conducive to interactive jobs, because there could be a long wait between job submission and resources becoming available.
* Interactive jobs allow users to enter commands and data on the command line, like when working on your laptop.
* **Use when:** Editing files, debugging code, exploring data, etc.
* **In our case**, we would use an interactive job on burst (as it gives instant compute access vs waiting for Greene allocation) to run inferencing once, to make sure our scripts work as intended before inferencing on thousands of samples with Greene.

#### Batch:
* Allows us to plan the sequence of commands to perform the actions we need in script form.
* **Use when**: Compute-heavy runs that need a longer duration of GPU resources, such as inferencing many samples.

## Writing and Submitting a Batch Job
A batch job consists of two things:
1. A .sbatch file, which contains the SBATCH instructions that describe the resources required & other info, for SLURM to read and allocate resources.
2. The script itself, which contains commands that don't require additional user interaction. 

### Making the .sbatch file
SLURM expects a file in a specific format in order to execute a job. See an example below:

**NOTE: This .sbatch file will run a CPU job. Keep reading for running a GPU job.**

_cpu_job.sbatch:_
```
# tells the shell how to execute the script, not related to SLURM 
!/bin/bash

#SBATCH --nodes=3                        # requests 3 compute servers
#SBATCH --ntasks-per-node=2              # runs 2 tasks on each server
#SBATCH --cpus-per-task=1                # uses 1 compute core per task

# How long we expect the job to take. If it takes longer than this, SLURM can kill it.
#SBATCH --time=1:00:00 

# How much memory we expect the job to use
#SBATCH --mem=2GB 

# Add these lines if you want to receive an email when the job ends
#SBATCH --mail-type=END
#SBATCH --mail-user=<your_email>@nyu.edu

# Name the job if you don't want it generated from the script name
# Also affects the name of the job as reported by the queue
#SBATCH --job-name=<job-name-here>

# Specify the output file name
#SBATCH --output=<output-file-name>.out 

# If you do not want to specify the filename, you can instead use
#SBATCH --output=slurm_%j.out

module purge                        # Ensures a clean running environment
module load python/intel/3.8.6      # Loads the module for the software we are using

#  Activate and run using your virtual environment
source /path/to/venv/bin/activate
srun /path/to/venv/bin/python ./<file-name>.py
```
* Any line beginning with #SBATCH is read by SLURM and used to determine queueing. If you want to comment out an SBATCH directive, use a second # e.g. `##SBATCH`
* Once the first non-comment, non-SBATCH-directive line is encountered, SLURM stops looking for SBATCH directives and runs the rest as a bash file.
* By default, the script will start running in the directory that you ran .sbatch from.
* For more information on additional SBATCH directives, see [here](https://sites.google.com/nyu.edu/nyu-hpc/training-support/general-hpc-topics/slurm-submitting-jobs?authuser=0#h.7xswxliwmidw).
* We can delay starting a job until another has finished. See above link "options for delaying starting a job" for more info.

### Submitting a job

Submit your job with the command:
```bash
sbatch <cpu_job>.sbatch
```
* It does not matter if you submit an .sbatch or .slurm file, both are functionally the same.

To check the status of your jobs:
```bash
squeue -u <net_id>
```

### The output file(s)
Your output will come in two parts, stdout and stderr, which by default are both written to the output file that you specified in the .sbatch script.
* **stdout:** Standard output. Used for normal program input. Captures results, print statements, or successful messages.
* **stderr:** Standard error. Ued for error messages. Captures warnings, exceptions, and runtime errors.
<br/>
#### What if I want to output stderr into a separate file?
Include one of the following lines in your .sbatch file:
```
#SBATCH --error=slurm_%j.err        # to automatically name it
or
#SBATCH --error=<file-name>.err     # to specify a filename
```
<br/>

#### When should I include print statements in .sbatch?
1. **Debugging SLURM script execution**. This scenario should be pretty rare. To do so, include the following line in your .sbatch script:
```bash
echo "Job completed"                # or some other statement
```

2. **Checking if files were outputted successfully.** Such as when we are outputting JSON files. Include at the very end of your .sbatch (after python run command):
```bash
# Confirm file exists
if [[ -f "<output-file-name>.json" ]]; then
    echo "JSON output generated successfully."
else
    echo "JSON output missing!" >&2
fi
```
<br/>

#### What if I want to run the job in a different working directory than my .sbatch file is in?
Include in .sbatch file:
```
#SBATCH --chdir=/path/to/output/dir
```
Your stderr and stdout will be output in this new directory unless you explicitly specify a different path with `#SBATCH --output` and `#SBATCH --error`.
<br/>

#### What if I want to move a file upon job completion?
This will be especially useful since we run jobs in `/scratch/` but want to store big data files in `/home/` or `/archive/`.
Include at the end of .sbatch:
```bash
mv <output>.json /home/$USER/...           # or another file location
```