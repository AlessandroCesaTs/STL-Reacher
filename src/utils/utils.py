import os
import multiprocessing

def get_num_cpus():
    # Check if running under SLURM
    if "SLURM_CPUS_PER_TASK" in os.environ:
        num_cpus = int(os.environ["SLURM_CPUS_PER_TASK"])
    else:
        # Fall back to the local machine's available CPUs
        num_cpus = multiprocessing.cpu_count()
    
    return num_cpus

