#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --no-requeue
#SBATCH --get-user-env

n_envs=$1

source bash_scripts/slurm_utils.sh  #source srun_if_on_slurm wich returns srun if i'm on a slurm environment

source env/bin/activate

echo "Start Training"

for i in {0..3}

    do srun_if_on_slurm python3 src/train.py --num_of_goals=3 --num_of_avoids=1 --total_timesteps=409600 --n_envs=$n_envs

    echo "Done Training"
done
echo "Done"
