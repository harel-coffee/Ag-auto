#!/bin/bash
#v#!/usr/bin/env Rscript
# job name

#PBS -N sensitivity_0
#PBS -l nodes=1:ppn=1,walltime=70:00:00
#PBS -l mem=10gb
#PBS -q hydro
#PBS -e sensitivity_0_error.txt
#PBS -o sensitivity_0_output.txt
#PBS -m abe
cd $PBS_O_WORKDIR

# First we ensure a clean running environment:
module purge

# Load modules (if needed)
module load R/R-3.2.2_gcc

/home/hnoorazar/cleaner_codes/drivers/sensitivity_shrunk_0.R

exit 0
