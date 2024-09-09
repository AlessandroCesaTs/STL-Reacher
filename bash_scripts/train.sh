#!/bin/bash
#SBATCH -J train
#SBATCH -o train.out
#SBATCH -p EPYC
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --hint=nomultithread
#SBATCH --time=1:00:00
#SBATCH --no-requeue
#SBATCH --get-user-env

output_path=${1:-$(pwd)}
num_of_iters=${2:-1}

source bash_scripts/slurm_utils.sh  #get slurm utils functions

if i_am_on_slurm; then
    total_timesteps=2048000
    n_steps=2048
    n_epochs=10
else
    total_timesteps=256
    n_steps=128
    n_epochs=2
fi

source env/bin/activate

for i in {0..${num_of_iters}}
do
    echo "Start Training"
    srun_if_on_slurm python3 -u src/train.py --num_of_goals=1 --num_of_avoids=1 --total_timesteps=${total_timesteps} --n_steps=${n_steps} --n_epochs=${n_epochs} --output_path=${output_path}
done 

echo "Job completed"
