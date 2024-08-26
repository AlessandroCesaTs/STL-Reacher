#!/bin/bash
#SBATCH -J test.sh
#SBATCH -o test.o
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --no-requeue
#SBATCH --get-user-env

source bash_scripts/slurm_utils.sh  #source srun_if_on_slurm wich returns srun if i'm on a slurm environment

source env/bin/activate

echo "Start Testing"

srun_if_on_slurm python3 src/test.py --num_of_goals=3 --num_of_avoids=1 --test_steps=400 --n_envs=2

echo "Done Testing"
