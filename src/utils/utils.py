import os
import multiprocessing
import shutil
import tempfile

def get_num_cpus():
    # Check if running under SLURM
    if "SLURM_CPUS_PER_TASK" in os.environ:
        num_cpus = int(os.environ["SLURM_CPUS_PER_TASK"])
    else:
        # Fall back to the local machine's available CPUs
        num_cpus = multiprocessing.cpu_count()
    
    return num_cpus


def copy_urdf_directory(urdf_dir):
    temp_dir = tempfile.mkdtemp()
    temp_urdf_dir=os.path.join(temp_dir,os.path.basename(urdf_dir))
    shutil.copytree(urdf_dir, temp_urdf_dir,dirs_exist_ok=True)
    return temp_urdf_dir

