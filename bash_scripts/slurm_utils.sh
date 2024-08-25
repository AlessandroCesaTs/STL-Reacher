#!/bin/bash

# Function to run a command either with or without srun based on if it's in a slurm environment
srun_if_on_slurm() {
    if command -v srun &> /dev/null; then
        # Running on SLURM 
        srun $@
    else
        # Not running on SLURM (likely running locally)
        $@
    fi
}