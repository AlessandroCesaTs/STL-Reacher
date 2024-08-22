#!/bin/bash
#SBATCH -j train_and_test.sh
#SBATCH -o train_and_test.o
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntask=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --no-requeue
#SBATCH --get-user-env

SCRIPT_DIR=$(dirname "$0")   #Get the directory where this script is located
source "$SCRIPT_DIR/slurm_utils.sh"  #source run_cmd wich returns srun if i'm on a slurm environment

source env/bin/activate

echo "Start train and test"

run_cmd python3 src/train_and_test.py

echo "Done"