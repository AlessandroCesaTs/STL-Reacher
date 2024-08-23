#!/bin/bash

source bash_scripts/slurm_utils.sh  #source run_cmd wich returns srun if i'm on a slurm environment

train_job_id=$(bash_or_sbatch bash_scripts/train.sh | awk '{print $4}')

test_job_id=$(bash_or_sbatch_with_dependency $train_job_id bash_scripts/test.sh)

echo "Submitted jobs $train_job_id, $test_job_id"