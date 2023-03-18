#!/bin/bash
#v#!/usr/bin/env Rscript

#PBS -N Merge_CMPOP_Files
#PBS -l nodes=1:ppn=1,walltime=11:59:00
#PBS -l mem=10gb
#PBS -q hydro
#PBS -e merge_CMPOP_error.txt
#PBS -o merge_CMPOP_output.txt
#PBS -M h.noorazar@yahoo.com
#PBS -m abe
cd $PBS_O_WORKDIR

# Ensure a clean running environment:
module purge

# Load modules (if needed)
module load R/R-3.2.2_gcc

/home/hnoorazar/cleaner_codes/drivers/merge_CMPOP_driver.R

exit 0
