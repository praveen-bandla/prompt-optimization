
# HPC High-level Guide

This document serves to give a brief guide on how to set up, and use the HPC provided by NYU. Mostly, will get rid of this in the final submission, but will keep this here for now.
By the end of this document, you should have a decent understand of HPC, and be comfortable with the setup of the HPC before any of the project specific setup. This includes: signing in and out, check resources quota, changing default directory, etc.

I AM A COMPLETE NOOB I COULD BE WRONG SORRY TYVM I TRIED MY BEST

<br>

## High-Level Guide of our HPC

### What is HPC?

High-Performance Computing (HPC) is a centralized computer far more powerful than regular laptops. Our HPC system, **Greene**, provides access to high-performance GPUs, large-scale storage, and distributed processors for running resource-intensive jobs.


### Nodes

In an HPC system, nodes are individual computers (servers) that work together to process jobs. We connect to the HPC through nodes. Types of nodes:

1. **Login Node**: The entry point where you connect to the HPC (via SSH). *This also works exactly like a local computer. In many ways, you treat it as identical*.
2. **Compute Nodes**: The machines that actually run your jobs. These have CPUs, GPUs, that we use. We do not directly interact with compute nodes. We send jobs via the job scheduler (SLURM - see more below)
3. **Burst Nodes**: Temporary compute nodes create on Google Cloud platform (GCP) when we use their resources instead


### Burst Nodes and GCP

Burst is the node that allows us to access compute resources from GCP. We connect to the burst node after connecting to the login node. We only need burst when using GCP resources. Considerations related to GCP:

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
- **Home (50GB)**: Main folder, is backed up. Has smaller file size limit and a limited number of files
- **Scratch (5TB)**: Not backed up. Has much larger file size limit. Jason recommended we just use scratch for everything.

Given that all our code will be backed up through repository, the only thing we need to worry about duplicating is our data files. We will back those up on Home, but use scratch otherwise for primary purpose.

<br>

## Connecting to HPC

If you are connecting from a remote location that is not on the NYU network (e.g. from your home), you will need to use the NYU VPN.
* See [Guide for Installing and Using VPN on Mac](https://www.nyu.edu/servicelink/KB0011175)
* See [Guide for Installing and Using VPN on Windows](https://www.nyu.edu/servicelink/KB0011177)

Once you have established an NYU VPN connection, you can proceed as if you were connected to the NYU network.

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

<br>

### Remote Explorer (**)

(Writing this section off of memory, I will sit with you to get this done). This section is super helfpul. It allows you to set up your HPC the same way you would use your VSCode for local files. By the end of this section, you should be able to navigate your file directory through HPC the way you would for local files.

#### Step 1: Download Remote - SSH

This is an extension on VSCode. Watch the little gif they have btw, super helpful.

#### Step 2: Set-up the SSH

Once you download the extension, you should see a little monitor icon pop-up in the sidebar. Open that and you should see an SSH toggle.

Once there, select the "+" next to "SSH" and enter: ```ssh {net_id}@greene.hpc.nyu.edu```

Then, it should prompt you to enter a config file for access. Use the one sourced from your local files.

#### Step 3: Access HPC

There should be a little icon at the bottom left. Click on it and it should give you an option to connect to host. Select the file directory you want and then we should be good.

MAKE SURE TO CLOSE REMOTE CONNECTION WHEN YOU'RE DONE. SHE GETS TRIGGERED WHEN YOU DONT
Click the the Remote Connection blue box (bottom left) and then choose close remote connection.


#### Troubleshooting Remote Explorer

Since making this guide, my remote explorer died twice. I don't have the exact message, but if its failing, you should see a warning error pop-up, and should read something like:

> *Could not establish connection to "greene.hpc.nyu.edu": Remote host key has changed, port forwarding is disabled.*

This is not a thorough guide on how to resolve it, but somethings that might work - I don't understand it fully, but more so documenting what worked for me.
If this happens, at a high-level, what you want to do is get rid of your Remote-SSH and restart it.

#### Step 1: Getting Rid of Existing SSH

First, remove the SSH key you have using (exactly):

```bash
ssh-keygen -R greene.hpc.nyu.edu
```

Then, make the corresponding changes to the config file. To do so, run `cmd + shift + P` (pallete, useful for accessing many meta options). Search for `Remote-SSH: Open SSH Configuration file`. Then, manually get rid of the saved SSH key-config that you see for greene hpc. Save the file and close.

#### Step 2: Reconnect to Greene via Terminal

Run your standard command to sign-in:

```bash
ssh {net_id}@greene.hpc.nyu.edu
```

Follow the process to sign-in.

#### Step 3: Add Remote-SSH once again

Do so with `cmd + shift + p` followed by `Remote-SSH: Add New SSH Host`

Follow the same procedure that you used to set-up remote explorer and you should be good? It worked perfectly the first time, but the second time didn't work super well. 




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

4. Once added, hit Enter. Then, type 'control + X' to exit the text editor. 

Now, for every successive login, you should open by default to scratch. If doing something in the `/home/{net_id}` file directory, navigate using `cd` and `ls`.

REMEMBER TO CLOSE THE REMOTE CONNECTION EVERYTIME YOU STEP OUT, OTHERWISE IT COULD KILL YOUR SSH KEY. I HAD TO RESTART MINE RN IT WAS SAD









