#!/bin/bash
#v#!/usr/bin/env Rscript
#PBS -N compare_files
#PBS -l nodes=1:ppn=1,walltime=11:59:00
#PBS -l mem=60gb
#PBS -q hydro
#PBS -e compare_error.txt
#PBS -o compare_output.txt
#PBS -m abe
cd $PBS_O_WORKDIR

# Ensure a clean running environment:
module purge

# Load modules (if needed)
module load R/R-3.2.2_gcc

/home/hnoorazar/cleaner_codes/drivers/compare_driver_table.R

exit 0
