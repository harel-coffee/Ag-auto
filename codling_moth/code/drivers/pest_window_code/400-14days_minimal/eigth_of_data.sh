#!/bin/bash
#!/usr/bin/env Rscript

#PBS -N eigth_of_data
#PBS -l nodes=1:ppn=1,walltime=99:00:00
#PBS -l mem=20gb
#PBS -q hydro
#PBS -e E_eigth_of_data.txt
#PBS -o O_eigth_of_data.txt
#PBS -m abe
cd $PBS_O_WORKDIR
# Ensure a clean running environment:
module purge
# Load modules (if needed)
module load R/R-3.2.2_gcc 0.25

./eigth_of_data.R

exit 0
