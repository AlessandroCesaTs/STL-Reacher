#!/bin/bash

for change_goals in "True" "False"
do
    mkdir /u/dssc/acesa000/STL-Reacher/output_goal_change_${change_goals}
    sbatch -J train_goal_change_${change_goals} -o train_goal_change_${change_goals}.out bash_scripts/train.sh "/u/dssc/acesa000/STL-Reacher/output_goal_change_${change_goals}" ${change_goals}
done