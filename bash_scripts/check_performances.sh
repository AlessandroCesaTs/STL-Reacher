csv_epyc_file='/u/dssc/acesa000/STL-Reacher/output_epyc/times.csv'
csv_thin_file='/u/dssc/acesa000/STL-Reacher/output_thin/times.csv'

echo "Cores, Time" >>$csv_epyc_file
echo "Cores, Time" >>$csv_thin_file


for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
do
    sbatch -p EPYC -J train_epyc_${i} -o train_epyc_${i}.out --cpus-per-task=$i bash_scripts/train.sh '/u/dssc/acesa000/STL-Reacher/output_epyc'
done

for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 20 24
do
    sbatch -p THIN -J train_thin_${i} -o train_thin_${i}.out --cpus-per-task=$i bash_scripts/train.sh '/u/dssc/acesa000/STL-Reacher/output_thin'
done