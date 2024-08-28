#!/bin/bash
#SBATCH -J train
#SBATCH -o train.out
#SBATCH -p EPYC
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --hint=nomultithread
#SBATCH --time=1:00:00
#SBATCH --no-requeue
#SBATCH --get-user-env

source bash_scripts/slurm_utils.sh  #source srun_if_on_slurm wich returns srun if i'm on a slurm environment

source env/bin/activate

echo "Start Training"
srun_if_on_slurm python3 -u src/train.py --num_of_goals=3 --num_of_avoids=1 --total_timesteps=4096
echo "Done"

