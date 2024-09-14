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

output_path=${1:-$(pwd)/output}

source bash_scripts/slurm_utils.sh  #get slurm utils functions

source env/bin/activate

echo "Start Plotting"
srun_if_on_slurm python3 src/plot_results.py --num_of_robustnesses=2 --test_runs=1

echo "Done Plotting"