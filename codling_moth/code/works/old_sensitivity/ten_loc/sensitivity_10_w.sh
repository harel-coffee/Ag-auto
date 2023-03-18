#!/bin/bash
#v#!/usr/bin/env Rscript
# job name

#PBS -N sensitivity_10_w
#PBS -l nodes=1:ppn=1,walltime=70:00:00
#PBS -l mem=10gb
#PBS -q hydro
#PBS -e sensitivity_10_e.txt
#PBS -o sensitivity_10_o.txt
#PBS -m abe
cd $PBS_O_WORKDIR

# First we ensure a clean running environment:
module purge

# Load modules (if needed)
module load R/R-3.2.2_gcc

./sensitivity_shrunk_10.R

exit 0
