#!/bin/bash

i_am_on_slurm() {
    # Check if SLURM commands are available
    if command -v scontrol &> /dev/null && command -v squeue &> /dev/null; then
        return 0  # True: SLURM commands are available;
    fi

    return 1  # False: Not on a SLURM-managed system
}

# Function to run a command either with or without srun based on if it's in a slurm environment
srun_if_on_slurm() {
    if i_am_on_slurm; then
        # Running on SLURM 
        srun $@
    else
        # Not running on SLURM (likely running locally)
        $@
    fi
}