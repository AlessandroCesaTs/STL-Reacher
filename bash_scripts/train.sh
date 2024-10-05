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

output_path=$(pwd)/outputs/${1:-output}
num_of_iters=${2:-1}

source bash_scripts/slurm_utils.sh  #get slurm utils functions

if i_am_on_slurm; then
    total_timesteps=153600
    n_steps=2000
    n_epochs=10
    max_steps=1000
else
    total_timesteps=256
    n_steps=128
    n_epochs=2
    max_steps=100
fi

source env/bin/activate

for i in {0..${num_of_iters}}
do
    echo "Start Training"
    srun_if_on_slurm python3 -u src/train.py --total_timesteps=${total_timesteps} --n_steps=${n_steps} --n_epochs=${n_epochs} --max_steps=${max_steps} --output_path=${output_path}
done 

echo "Job completed"
