#!/bin/bash
#v#!/usr/bin/env Rscript
#PBS -N take_samples
#PBS -l nodes=1:ppn=1,walltime=11:59:00
#PBS -l mem=10gb
#PBS -q hydro
#PBS -e take_samples_error.txt
#PBS -o take_samples_output.txt
#PBS -m abe
cd $PBS_O_WORKDIR

# Ensure a clean running environment:
module purge

# Load modules (if needed)
module load R/R-3.2.2_gcc

./take_samples.R

exit 0
