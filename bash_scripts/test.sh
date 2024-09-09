#!/bin/bash
#SBATCH -J test
#SBATCH -o test.out
#SBATCH -p EPYC
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --hint=nomultithread
#SBATCH --time=1:00:00
#SBATCH --no-requeue
#SBATCH --get-user-env

output_path=${1:-$(pwd)}

source bash_scripts/slurm_utils.sh  #source srun_if_on_slurm wich returns srun if i'm on a slurm environment

source env/bin/activate

echo "Start Testing"

srun_if_on_slurm python3 src/test.py --num_of_goals=2 --output_path=${output_path}

echo "Done Testing"
