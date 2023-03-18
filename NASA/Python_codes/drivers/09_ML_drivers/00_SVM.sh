#!/bin/bash

# ----------------------------------------------------------------
# Configure PBS options
# ----------------------------------------------------------------
## Define a job name
#PBS -N mergeWides_moreBands

## Define compute options
#PBS -l nodes=1:ppn=1
#PBS -l mem=30gb
#PBS -l walltime=06:00:00
##PBS -l walltime=99:00:00
##PBS -q hydro

## Define path for output & error logs
#PBS -k o

#PBS -e /home/hnoorazar/NASA/03_01_widen_MoreBands/error/mergeWides_moreBands_e
#PBS -o /home/hnoorazar/NASA/03_01_widen_MoreBands/error/mergeWides_moreBands_o

## Define path for reporting
##PBS -M h.noorazar@yahoo.com
#PBS -m abe

# ----------------------------------------------------------------
# Start the script itself
# ----------------------------------------------------------------
module purge
module load gcc/7.3.0
module load python/3.7.1/gcc/7.3.0

cd /home/hnoorazar/NASA/03_01_widen_MoreBands
   

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

python3 ./03_01_merge_train_moreBands_Wide.py






