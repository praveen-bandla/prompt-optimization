
# HPC High-level Guide

This document serves to give a brief guide on how to set up, and use the HPC provided by NYU. Mostly, will get rid of this in the final submission, but will keep this here for now.
By the end of this document, you should have a decent understand of HPC, and be comfortable with the setup of the HPC before any of the project specific setup. This includes: signing in and out, check resources quota, changing default directory, etc.

I AM A COMPLETE NOOB I COULD BE WRONG SORRY TYVM I TRIED MY BEST


## High-Level Guide of our HPC

### What is HPC?

High-Performance Computing (HPC) is a centralized computer far more powerful than regular laptops. Our HPC system, **Greene**, provides access to high-performance GPUs, large-scale storage, and distributed processors for running resource-intensive jobs.


### Nodes

In an HPC system, nodes are individual computers (servers) that work together to process jobs. we connect to the HPC through nodes. Types of nodes:

1. **Login Node**: The entry point where you connect to the HPC (via SSH). *This also works exactly like a local computer. In many ways, you treat it as identical*.
2. **Compute Nodes**: The machines that actually run your jobs. These have CPUs, GPUs, that we use. We do not directly interact with compute nodes. We send jobs via the job scheduler (SLURM - see more below)
3. **Burst Nodes**: Temporary compute nodes create on Google Cloud platform (GCP) when we use their resources instead


### Burst Nodes and GCP

Bust is the node that allows us to access compute resources from GCP. We connect to the burst node after connecting to the login node. We only need burst when using GCP resources. Considerations related to GCP:

1. **Burst runs a separate SLURM cluster on GCP, not on Greene's on-premise (on-prem) HPC**. This is associated with our credentials - we are approved through NLU.
2. **We must copy data from Greene to the Burst instance before running Jobs**: The files that we will store in Green's directory is not automatically accessible via burst (see later for file directory).
3. **Burst is useful when we need resources immediately**: Access is instant and we don't have to wait in queue (see later for job scheduling).

For Greene, we are not resource constrained. Once we have approval for resources, we are good to use however much we need. For our project **I think we are best using just Greene and not burst**. [Jason also said he never uses Burst, just Greene directly].


### SLURM Job Scheduling

SLURM is the job scheduler used to manage compute resources on our HPC. It allocates resources and queues jobs. Instead of running GPU-related code directly on login nodes, we submit jobs to SLURM, which assigns them to available compute nodes.

Key Concepts:<br>
*Most of these we will standardize for tasks* but nonetheless:
- **Jobs**: Tasks submitted to SLURM for execution. These are most often python scripts
- **Nodes**: Compute units where jobs run
- **Partitions**: Group of nodes with specific resource configurations
- **Resource Requests**: We specify CPU/GPUs needed, memory, time. Based on our requirements, it schedules it for us (based on priority and resource availability).

(Not 100% Sure): We do not necessarily need to specify nodes. By default, SLURM assigns jobs to available nodes based on resource requests. However, we can specify nodes or partitions we need specific hardware or configurations. I think we were assigned a partition for NLU - not sure if we need to use it.

### File Directory

We have 3 categories of file storage on the HPC. One of them is `archive` which we will ignore for now. The other two:
- **Home (50GB)**: Main folder, is backed up. Has smaller file size limite and a limited number of files
- **Scratch (5TB)**: Not backed up. Has much larger file size limit. Jason recommended we just use scratch for everything.

Given that we all code backed up through repository, the only thing we need to worry about duplicating is our data files. We will back those up on Home, but use scratch otherwise for primary purpose.


## Setting up HPC

For all commands below, use bash terminal in VSCode (unless otherwise specified).

### Signing In

```bash
ssh <net_id>@greene.hpc.nyu.edu
```

Then, enter your password and you are connected. When you are done, enter:

```bash
exit
```
to close the connection.

### Error Re-signing In? (Optional read, if you run into issue)
Sometimes, when you login, logout, and then try to login again, it will throw an error, something along the lines of:

```bash
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@    WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED!     @
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
IT IS POSSIBLE THAT SOMEONE IS DOING SOMETHING NASTY!
Someone could be eavesdropping on you right now (man-in-the-middle attack)!
It is also possible that a host key has just been changed.
```

Use this to fix:

```bash
ssh-keygen -R greene.hpc.nyu.edu
```

### Checking Quota

As mentioned earlier, you have file sizes available per type of directory. Use this command to check:

```bash
myquota
```

### File Directory System

I'll go over everything to be thorough so feel free to skip parts you are comfortable.
Every Bash command runs from a file directory. You can change the file directory as needed, here are some commands that are helpful to know to change the directory:

```bash
ls  # list current directory
cd subfolder  # set current directory to subfolder (note: no string)
cd ..   # set cd to parent folder
cd ../..    # set cd to parent folder's parent folder
pwd     # print current working directory
```

When you first login to your HPC, you are in `/home/{your_netid}`. We will primarily be working in the scratch folder which is `/scratch/{your_netid}`. Instead of having to navigate each time to the scratch folder, there is a way to change the default folder you open when signing into HPC. See below

#### Change Default File Directory

A `.bashrc` file stands for **Bash Run Commands**. When you log into HPC, this file will run automatically.  To change the default fd, we edit the file:

(First log into HPC)

1. Open the `.bashrc` file to edit using:

```bash
nano ~/.bashrc
```

2. View the file, should look something like this:

```bash
# Source global definitions
if [ -f /etc/bashrc ]; then
        . /etc/bashrc
fi

# User specific environment
if ! [[ "$PATH" =~ "$HOME/.local/bin:$HOME/bin:" ]]
then
    PATH="$HOME/.local/bin:$HOME/bin:$PATH"
fi
export PATH

# Uncomment the following line if you don't like systemctl's auto-paging feature:
# export SYSTEMD_PAGER=

# User specific aliases and functions
if [ -d ~/.bashrc.d ]; then
        for rc in ~/.bashrc.d/*; do
```

3. To the very bottom of the file, add the following code:

```bash
cd /scratch/{your_netid}
```

Now, for every successive login, you should open by default to scratch. If doing something in the `/home/{net_id}` file directory, navigate using `cd` and `ls`.










