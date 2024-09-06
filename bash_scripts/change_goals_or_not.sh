#!/bin/bash

for change_goals in "True" "False"
do
    if command -v sbatch &> /dev/null; then
        train_id=$(sbatch -J train_goal_change_${change_goals} -o train_goal_change_${change_goals}.out bash_scripts/train.sh "$(pwd)/output_goal_change_${change_goals}" ${change_goals}| awk '{print $4}')
        sbatch --dependency=afterok:$train_job_id -J test_goal_change_${change_goals} -o test_goal_change_${change_goals}.out bash_scripts/test.sh "$(pwd)/output_goal_change_${change_goals}" ${change_goals}
    else
        bash_scripts/train.sh "$(pwd)/output_goal_change_${change_goals}" ${change_goals}
        bash_scripts/test.sh "$(pwd)/output_goal_change_${change_goals}" ${change_goals}
    fi
done