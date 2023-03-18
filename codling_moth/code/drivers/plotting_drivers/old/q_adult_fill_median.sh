#!/bin/bash
#v#!/usr/bin/env Rscript

#PBS -N adult_fill_median_45
#PBS -l nodes=1:ppn=1,walltime=99:00:00
#PBS -l mem=30gb
#PBS -q hydro
#PBS -e adult_median_45_E.txt
#PBS -o adult_median_45_o.txt
#PBS -m abe
cd $PBS_O_WORKDIR

# Ensure a clean running environment:
module purge

# Load modules (if needed)
module load R/R-3.2.2_gcc

./d_adult_DoY_fill_median.R rcp45

exit 0