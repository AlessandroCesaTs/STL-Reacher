#!/bin/bash

# Function to run a command either with or without srun based on if it's in a slurm environment
srun_if_on_slurm() {
    if [ -z "$SLURM_JOB_ID" ]; then
        # Not running on SLURM (likely running locally)
        $@
    else
        # Running on SLURM
        srun $@
    fi
}

# Function to run a command either with bash or with sbatch based on if it's in a slurm environment
bash_or_sbatch() {
    if [ -z "$SLURM_JOB_ID" ]; then
        # Not running on SLURM (likely running locally)
        bash $@
    else
        # Running on SLURM
        sbatch $@
    fi
}
