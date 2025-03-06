# Set-up Guide for Our Project

Ty for reading through the first guide, welcome to the second equally poorly written guide.

I'd like to thank my dear Chatgpt, becca, becca's climbing friend, reddit, and whoever invited caffeine for getting me through all this hpc mess.

By this point, you should have an understanding of how HPC works along with working knowledge of how to interact with it, set it up. This document will go into project specific set-up, including repository usage, environment setup. It will also include an overview of branch usage (might make a dedicated section for branches but might be too much to include - will focus on giving a good enough guide for you to then google/chatgpt specifics).


## GitHub Cloning

This is an important step, it sets-up permanent access between your HPC and GitHub repository. For any work you do on the HPC that includes the use of GitHub or any Git-based software, you will likely do the same thing. This process should be relatievly similar to what we did in Lab for Dataproc. Here is the process:

### Step 1: Check for Existing SSH Keys

First, sign onto your HPC.
Then, run (exactly):

```bash
ls -al ~/.ssh
```
You should see either `id_rsa` or `id_ed25519`. If they exist, you can move to the next step. I'm not sure what to do if those files don't exist but lmk so we can find out how to create them.

### Step 2: Generate a new SSH Key

On a new terminal line, run:
```bash
ssh-keygen -t ed25519 -C "your_github_email@example.com"
```

You should be prompted to specify the file location. Hit enter so that it uses default. Then, you should be prompted for passphrase. Hit enter again so that it uses no passphrase (looked it up, you don't really need a passphrase, good enough security).

### Step 3: Add SSH Key

The previous step generates the key. Now you need to add it to the SSH agent. To do so, just run (exactly):

```bash
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
```

### Step 4: Get the Public Key

Run:

```bash
cat ~/.ssh/id_ed25519.pub
```

This should be in the format: `ssh-ed25519 XXXX your_email@nyu.edu` or something like that. Copy the entire thing.


### Step 5: Add SSH Key to GitHub

Go to your SSH and GPG Keys in Settings. Click on New SSH Key and add what you copied into the 'key' field. Title should be something like 'NYU - HPC'


### Step 6: Testing the SSH Connection

First, verify that the SSH connection worked. Run (exactly):

```bash
ssh -T git@github.com
```

You should see something like:

```bash
Hi <username>! You've successfully authenticated, but GitHub does not provide shell access.
```

(idk what the last part means)

### Step 7: Clone the repository

You should have set up the authentication by this point. Now, all you need to do is clone the repository. Normally, I clone with the `HTTPs` toggle, but here, you should clone with the `SSH` toggle.

Should be something like:

```bash
git clone git@github.com:<your-username>/<your-repository>.git
```

That should be everything! When you next SSH in, you should be able to access the repository. I'm actually writing this on the HPC so yipee (send help pls too many bash commands i dont have the compute).

<br>

## Environment Set-up

### Overview

One of the things that is different between your local machine and HPC is admin access. Normally, you download all packages you need into your base environment. Because you run everything in one place, your environment is shared across the system. However, here, when running jobs, we have to send the computation (jobs as scripts) to a different node. The environment is not shared, so we may run into package problems. Thus, our best bet is to create a **Virtual Environment**. We could also create a singularity - which is essentially a more granular, contained, and expansive structure that we can store a venv in. But its overkill, we likely don't need it. Jason also said we can just do a venv when I spoke to him.

### How Venvs work

Venv - virtual enviornments, are a python-specific tool used to create isolated environments for Python Projects, where we can manage Python dependencies without affecting the global Python installation on system.

### High-level Process Explanation For Project

We each have a venv that we store on our repository clones. These files exist in our local copies and not actually in the main (added to gitignore). Whenever we add new packages, we update the `requirements.txt` file (a file that contains all packages we use). Thus, whenever we pull main/branch, we update our venv to include any new packages that we used by downloading the differences in packages between what we already had venv and any new packages in `requirements.txt`. There are commands to simplify everything dw (I worried).

Further notes as it relates to the project:
- The venv is **NOT** tied to a specific branch and will exist in your repository once created. It is independent of Git branches and persists across branch switches.
- It is not affected by resetting your branch. Since the venv is not tracked by Git, switching or resetting branches won't modify your venv.
- If you load a different state of the project (i.e., switch to a different branch that has a different set of dependencies), you should refresh your venv to ensure it matches the state of the project in that branch (by installing the packages listed in `requirements.txt`).

### Setting up Your Very Own VENV

#### Installing Pip in Your Login Node

Given how much I need pip to get through life, I installed in my Login Node - using the direct method. I recommend you do the same before anything. Here are the commands. Run (exactly):

```bash
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python3 get-pip.py
```

#### Creating a Venv

First, make sure you are within `prompt-optimization` directory. Then run:

```bash
python3 -m venv venv-{your_name}
```
This adds a venv to the file directory (should pop up on the left).

#### Adding Virtual Environment to `.gitignore`

Open the `.gitignore` file and then add:
```bash
venv-{your_name}/
```
This will tell Git to ignore the `venv-praveen` directory, meaning it won’t be committed to the repository.

#### Activate the Virtual Environment

After creating the virtual environment, you activate it using:

```bash
source venv-praveen/bin/activate
```

#### Install Dependencies from `requirements.txt`

Once `pip` is set up, you should install the dependencies listed in `requirements.txt`. Run:

```bash
pip install -r requirements.txt
```

#### Deactivating the Venv

When you're done working, deactivate the venv by running

```base
deactivate
```

### Adding Packages to the Environment

First, make sure your venv is activated. Then, install the needed package using. Run:

```bash
pip install {package_name}
```

Once you install the package, you need to update the `requirements.txt` by running:

```bash
pip freeze > requirements.txt
```

### Updating your venv

If there are changes to `requirements.txt`, you update by running:
(make sure your venv is activated first)
```bash
pip install -r requirements.txt
```

<br>

## Git Branches

I will divide up this subsection into a conceptual guide and a usage section.

### Conceptual Guide

A **branch** in Git is essentially a linked copy derived from a specific snapshot of your codebase. It represents an independent line of development within your project derived from a specific root directory. Like literal tree branches. Branches allow you to make changes independent of the main project - useful for working on different parts of the project separately, to ensure that a bug in a branch doesn't screw everything in main, etc.

#### When Do We Create a Branch



