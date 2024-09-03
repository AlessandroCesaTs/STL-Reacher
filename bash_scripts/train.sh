#!/bin/bash
#SBATCH -J train
#SBATCH -o train.out
#SBATCH -p EPYC
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=1:00:00
#SBATCH --no-requeue
#SBATCH --get-user-env

#output_path=$1

source bash_scripts/slurm_utils.sh  #source srun_if_on_slurm wich returns srun if i'm on a slurm environment

source env/bin/activate

for i in {0..0}
do
    echo "Start Training"
    srun_if_on_slurm python3 -u src/train.py --num_of_goals=3 --num_of_avoids=1 --total_timesteps=2048000
done 

echo "Job completed"

