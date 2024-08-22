#!/bin/bash

# Function to run a command either with or without srun based on the environment
run_cmd() {
    if [ -z "$SLURM_JOB_ID" ]; then
        # Not running on SLURM (likely running locally)
        $@
    else
        # Running on SLURM
        srun $@
    fi
}
