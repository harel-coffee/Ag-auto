#!/bin/bash
#SBATCH --partition=kamiak
#SBATCH --constraint=cascadelake
##SBATCH --partition=rajagopalan
#SBATCH --requeue
#SBATCH --job-name=indeks_smooth_type_batch_no # Job Name
#SBATCH --time=1-00:00:00    # Wall clock time limit in Days-HH:MM:SS
#SBATCH --mem=10GB 
#SBATCH --nodes=1            # Node count required for the job
#SBATCH --ntasks-per-node=1  # Number of tasks to be launched per Node
#SBATCH --ntasks=1           # Number of tasks per array job
#SBATCH --cpus-per-task=1    # Number of threads per task (OMP threads)
####SBATCH --array=0-30000

###SBATCH -k o
#SBATCH --output=/home/h.noorazar/NASA/trend/clean_plots_4_DL/error/indeks_smooth_type_batch_no_pre2008.o
#SBATCH --error=/home/h.noorazar/NASA/trend/clean_plots_4_DL/error/indeks_smooth_type_batch_no_pre2008.e
echo
echo "--- We are now in $PWD ..."
echo

## echo "I am Slurm job ${SLURM_JOB_ID}, array job ${SLURM_ARRAY_JOB_ID}, and array task ${SLURM_ARRAY_TASK_ID}."

# Load R on compute node
# module load r/4.1.0

## module purge         # Kamiak is not similar to Aeolus. purge on its own cannot be loaded. 
                        # Either leave it out or add "module load StdEnv". Lets see if this works. (Feb 22.)
## module load StdEnv

module load gcc/7.3.0
# module load python3/3.5.0
# module load python3/3.7.0
module load anaconda3

# pip install shutup
# pip install dtaidistance
# pip install tslearn

# ----------------------------------------------------------------
# Gathering useful information
# ----------------------------------------------------------------
echo "--------- environment ---------"
env | grep PBS

echo "--------- where am i  ---------"
echo WORKDIR: ${PBS_O_WORKDIR}
echo HOMEDIR: ${PBS_O_HOME}

echo Running time on host `hostname`
echo Time is `date`
echo Directory is `pwd`

echo "--------- continue on ---------"

# ----------------------------------------------------------------
# Run python code for matrix
# ----------------------------------------------------------------

python /home/h.noorazar/NASA/trend/clean_plots_4_DL/trend_cleanPlot_4_DL_pre2008.py indeks smooth_type batch_no

echo
echo "----- DONE -----"
echo

exit 0
