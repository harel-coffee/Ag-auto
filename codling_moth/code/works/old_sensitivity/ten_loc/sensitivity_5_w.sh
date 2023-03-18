#!/bin/bash
#v#!/usr/bin/env Rscript
# job name

#PBS -N sensitivity_5_w
#PBS -l nodes=1:ppn=1,walltime=70:00:00
#PBS -l mem=10gb
#PBS -q hydro
#PBS -e sensitivity_5_e.txt
#PBS -o sensitivity_5_o.txt
#PBS -m abe
cd $PBS_O_WORKDIR

# First we ensure a clean running environment:
module purge

# Load modules (if needed)
module load R/R-3.2.2_gcc

./sensitivity_shrunk_5.R

exit 0
