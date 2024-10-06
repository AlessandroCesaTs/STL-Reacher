#!/bin/bash
#SBATCH -J test
#SBATCH -o test.out
#SBATCH -p EPYC
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --hint=nomultithread
#SBATCH --time=1:00:00
#SBATCH --no-requeue
#SBATCH --get-user-env

change_target=${1:-'--change-target'}
output_path=$(pwd)/outputs/${2:-output}

source bash_scripts/slurm_utils.sh  #get slurm utils functions

if i_am_on_slurm; then
    max_steps=1000
    test_runs=5
else
    max_steps=10
    test_runs=1
fi

source bash_scripts/slurm_utils.sh  #source srun_if_on_slurm wich returns srun if i'm on a slurm environment

source env/bin/activate

echo "Start Testing"

srun_if_on_slurm python3 src/test.py --num_of_goals=2 --max_steps=${max_steps} --test_runs=${test_runs} --output_path=${output_path} ${change_target}

echo "Done Testing"