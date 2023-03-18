#!/bin/bash
#v#!/usr/bin/env Rscript

#PBS -N merge_CM_85
#PBS -l nodes=1:ppn=1,walltime=70:00:00
#PBS -l mem=20gb
#PBS -q hydro
#PBS -e merge_CM_85_e.txt
#PBS -o merge_CM_85_o.txt
#PBS -m abe
cd $PBS_O_WORKDIR

# Ensure a clean running environment:
module purge

# Load modules (if needed)
module load R/R-3.2.2_gcc

/home/hnoorazar/cleaner_codes/drivers/merge_driver.R rcp85 CM

exit 0
