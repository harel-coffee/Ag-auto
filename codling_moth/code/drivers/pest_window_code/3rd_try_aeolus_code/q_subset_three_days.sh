#!/bin/bash
#!/usr/bin/env Rscript

#PBS -N sub_3_days
#PBS -l nodes=1:ppn=1,walltime=1:00:00
#PBS -l mem=20gb
#PBS -q hydro
#PBS -e E_sub_3_days.txt
#PBS -o O_sub_3_days.txt
#PBS -m abe
cd $PBS_O_WORKDIR
# Ensure a clean running environment:
module purge
# Load modules (if needed)
module load R/R-3.2.2_gcc 

./d_subset_three_days.R

exit 0
