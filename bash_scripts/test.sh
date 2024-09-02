#!/bin/bash
#SBATCH -J test
#SBATCH -o test.out
#SBATCH -p EPYC
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBTAHC --cpus-per-task=10
#SBATCH --time=1:00:00
#SBATCH --no-requeue
#SBATCH --get-user-env

source bash_scripts/slurm_utils.sh  #source srun_if_on_slurm wich returns srun if i'm on a slurm environment

source env/bin/activate

echo "Start Testing"

srun_if_on_slurm python3 src/test.py --num_of_goals=3 --num_of_avoids=1

echo "Done Testing"
