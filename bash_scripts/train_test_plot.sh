#!/bin/bash

source bash_scripts/slurm_utils.sh  #get slurm functions

if i_am_on_slurm; then
    #I'm on slurm
    train_job_id=$(sbatch bash_scripts/train.sh | awk '{print $4}')
    test_job_id=$(sbatch --dependency=afterok:$train_job_id bash_scripts/test.sh | awk '{print $4}')
    plot_job_id=$(sbatch --dependency=afterok:$test_job_is bash_scripts/plot.sh | awk '{print $4}')
    echo "Submitted jobs $train_job_id, $test_job_id, $plot_job_id"
else
    #bash_scripts/train.sh
    bash_scripts/test.sh
    bash_scripts/plot.sh
fi
