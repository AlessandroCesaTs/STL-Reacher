for i in 1 2 4 8 16 17 32 64 128
do
    sbatch -J train_${i} -o train_${i}.out --cpus-per-task=$i bash_scripts/train.sh
done