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

double=${1:-'--double'}
plot_test=${2:'--plot_test'}
output_path=$(pwd)/outputs/${3:-output}


source bash_scripts/slurm_utils.sh  #get slurm utils functions

source env/bin/activate

echo "Start Plotting"

srun_if_on_slurm python3 src/plot_results.py ${double} ${plot_test} --output_path=${output_path}

echo "Done Plotting"