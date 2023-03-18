#!/bin/bash
# #SBATCH --partition=rajagopalan   # Partition (like a queue in PBS)
#SBATCH --partition=rajagopalan,cahnrs,cahnrs_gpu,kamiak
#SBATCH --requeue
#SBATCH --job-name=R_example # Job Name
#SBATCH --output=R_%j.out
#SBATCH --error=R_%j.err
#SBATCH --time=1-00:00:00    # Wall clock time limit in Days-HH:MM:SS
#SBATCH --nodes=1            # Node count required for the job
# #SBATCH --ntasks-per-node=1  # Number of tasks to be launched per Node
#SBATCH --ntasks=1           # Number of tasks per array job
#SBATCH --cpus-per-task=1    # Number of threads per task (OMP threads)
#SBATCH --array=0-563        # Array index

cd "$HOME"
echo
echo "--- We are now in $PWD, running an R script ..."
echo

# Load R on compute node
module load r/3.6.3


n=10 # grid locations per task

echo "I am Slurm job ${SLURM_JOB_ID}, array job ${SLURM_ARRAY_JOB_ID}, and array task ${SLURM_ARRAY_TASK_ID}."
Rscript --vanilla /home/vladyslav.oles/hardiness/hardiness_driver.R obs "$(( SLURM_ARRAY_TASK_ID * n ))" "$(( SLURM_ARRAY_TASK_ID * n + n - 1 ))"
