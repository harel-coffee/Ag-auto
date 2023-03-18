#!/bin/bash
#v#!/usr/bin/env Rscript

#PBS -N generations_85
#PBS -l nodes=1:ppn=1,walltime=99:00:00
#PBS -l mem=15gb
#PBS -q hydro
#PBS -e E_gen_85.txt
#PBS -o gen_85.txt
#PBS -m abe
cd $PBS_O_WORKDIR

# Ensure a clean running environment:
module purge

# Load modules (if needed)
module load R/R-3.2.2_gcc

/home/hnoorazar/cleaner_codes/drivers/generations_driver.R rcp85

exit 0
