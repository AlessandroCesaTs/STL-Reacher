#sbatch -J 1_env_0_gpu -o 1_env_0_gpu.out --ntasks=1 --gres=gpu:0 bash_scripts/train.sh 1
sbatch -J 2_env_0_gpu -o 2_env_0_gpu.out --ntasks=1 --cpus-per-task=2 --gres=gpu:0 bash_scripts/train.sh 2
#sbatch -J 1_env_1_gpu -o 1_env_1_gpu.out --ntasks=1 --gres=gpu:1 bash_scripts/train.sh 1
sbatch -J 2_env_1_gpu -o 2_env_1_gpu.out --ntasks=1 --cpus-per-task=2 --gres=gpu:1 bash_scripts/train.sh 2