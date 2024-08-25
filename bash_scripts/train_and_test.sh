#!/bin/bash

if command -v sbatch &> /dev/null; then
    #I'm on slurm
    train_job_id=$(sbatch bash_scripts/train.sh | awk '{print $4}')
    test_job_id=$(sbatch --dependency=afterok:$train_job_id bash_scripts/tesr.sh | awk '{print $4}')
    echo "Submitted jobs $train_job_id, $test_job_id"
else
    bash_scripts/train.sh
    bash_scripts/test.sh
fi
