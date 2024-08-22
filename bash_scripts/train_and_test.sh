#!/bin/bash

SCRIPT_DIR=$(dirname "$0")   #Get the directory where slurm scripts are located
source "$SCRIPT_DIR/slurm_utils.sh"  #source run_cmd wich returns srun if i'm on a slurm environment

bash_or_sbatch $SCRIPT_DIR/train.sh

bash_or_sbatch $SCRIPT_DIR/test.sh