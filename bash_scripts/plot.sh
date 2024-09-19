#!/bin/bash
#SBATCH -J plot
#SBATCH -o plot.out
#SBATCH -p EPYC
#SBATCH --nodes=1
#SBATCH --mem=8G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0:10:00
#SBATCH --no-requeue
#SBATCH --get-user-env

output_path=$(pwd)/outputs/${1:-output}

source bash_scripts/slurm_utils.sh  #get slurm utils functions

if i_am_on_slurm; then
    test_runs=5
else
    test_runs=1
fi


source env/bin/activate

echo "Start Plotting"

srun_if_on_slurm python3 src/plot_results.py --num_of_robustnesses=2 --test_runs=${test_runs} --output_path=${output_path} --no-plot_test

echo "Done Plotting"