#!/bin/bash

# ----------------------------------------------------------------
# Configure PBS options
# ----------------------------------------------------------------
## Define a job name
#PBS -N plt_2021

## Define compute options
#PBS -l nodes=1:ppn=1
#PBS -l mem=20gb
#PBS -l walltime=12:00:00
##PBS -q batch
#PBS -q hydro

## Define path for output & error logs
#PBS -k o
#PBS -e /home/hnoorazar/2021_doubleCroppedFieldDetection/00_first_plot_attempt/error/plt_all_jut_2021_e
#PBS -o /home/hnoorazar/2021_doubleCroppedFieldDetection/00_first_plot_attempt/error/plt_all_jut_2021_o

## Define path for reporting
#PBS -M h.noorazar@yahoo.com
#PBS -m abe

# ----------------------------------------------------------------
# Start the script itself
# ----------------------------------------------------------------
module purge
module load gcc/7.3.0
module load python/3.7.1/gcc/7.3.0

cd /home/hnoorazar/2021_doubleCroppedFieldDetection/00_first_plot_attempt

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
python3 ./00_first_plot_attempt_just_2021.py


