#!/bin/bash

output_directory=${1:-output}

source bash_scripts/slurm_utils.sh  #get slurm functions

if i_am_on_slurm; then
    #I'm on slurm
    train_job_id=$(sbatch --output=$(pwd)/outputs/${output_directory}/train/train.out bash_scripts/train.sh ${output_directory}| awk '{print $4}')
    test_job_id=$(sbatch --dependency=afterok:$train_job_id --output=$(pwd)/outputs/${output_directory}/test/test.out bash_scripts/test.sh ${output_directory}| awk '{print $4}')
    plot_job_id=$(sbatch --dependency=afterok:$test_job_id --output=$(pwd)/outputs/${output_directory}/test/plot.out bash_scripts/plot.sh ${output_directory}| awk '{print $4}')
    echo "Submitted jobs $train_job_id, $test_job_id, $plot_job_id"
else
    bash_scripts/train.sh
    bash_scripts/test.sh
    bash_scripts/plot.sh
fi
