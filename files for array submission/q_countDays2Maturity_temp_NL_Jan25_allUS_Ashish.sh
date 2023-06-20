#!/bin/bash
#SBATCH --partition=rajagopalan,stockle,cahnrs,cahnrs_bigmem,cahnrs_gpu,kamiak --account rajagopalan #cahnrs,cahnrs_bigmem,cahnrs_gpu
#SBATCH --requeue
#SBATCH --job-name=spi85bcccsm11 # Job Name
#SBATCH --time=2-12:00:00    # Wall clock time limit in Days-HH:MM:SS
#SBATCH --mem=06GB 
#SBATCH --nodes=1            # Node count required for the job
#SBATCH --ntasks-per-node=1  # Number of tasks to be launched per Node
#SBATCH --ntasks=1           # Number of tasks per array job
#SBATCH --cpus-per-task=1    # Number of threads per task (OMP threads)
#SBATCH --output=/data/project/agaid/fabio_scarpare/F_and_V/drivers/01_countDays_toReachMaturity/error/spinach_bcccsm11_85_array_%j.o
#SBATCH --error=/data/project/agaid/fabio_scarpare/F_and_V/drivers/01_countDays_toReachMaturity/error/spinach_bcccsm11_85_array_%j.e
##SBATCH --array=1-7208%150 			# Number of concurrent jobs %
#SBATCH --array=1-530%100

echo
echo "--- We are now in $PWD, running an R script ..."
echo

module load r/4.1.0

line_N=$( awk "NR==$SLURM_ARRAY_TASK_ID" input_for_fabio_test.txt)
#veg_type=$( echo "$line_N" | cut -d " " -f 1 )  
start_doy=$( echo "$line_N" | cut -d " " -f 1 ) 
region_part=$( echo "$line_N" | cut -d " " -f 2 )  

Rscript --vanilla /data/project/agaid/fabio_scarpare/F_and_V/drivers/01_countDays_toReachMaturity/d_countDays_to_maturity_NL_Jan25_allUS_Ashish.R ${start_doy} ${region_part}  

echo
echo "----- DONE -----"
echo

exit 0
