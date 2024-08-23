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

# Function to run a command either with bash or with sbatch based on if it's in a slurm environment
bash_or_sbatch() {
    if command -v sbatch &> /dev/null; then
        # Running on SLURM 
        sbatch $@
    else
        # Not running on SLURM (likely running locally)
        bash $@
    fi
}


# Function to run a command either with dependency flag based on if it's in a slurm environment
bash_or_sbatch_with_dependency() {
    
    if command -v sbatch &> /dev/null; then
        # If SLURM is available, add the --dependency flag
        dependency_flag="--dependency=afterok:$1"
        shift # Shift the first argument (dependency job ID) if it exists
        sbatch $dependency_flag "$@"
    else
        # If SLURM is not available, just run with bash
        shift
        bash "$@"
    fi
}
