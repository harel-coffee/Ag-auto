#!/bin/bash

#PBS -N count_rows
#PBS -l nodes=1:ppn=1,walltime=99:00:00
#PBS -l mem=12gb
#PBS -q hydro
#PBS -e E_count_rows.txt
#PBS -o O_count_rows.txt
#PBS -m abe
cd $PBS_O_WORKDIR
# Ensure a clean running environment:
module purge
# Load modules (if needed)
#module load R/R-3.2.2_gcc

./d_count_rows.R

exit 0
