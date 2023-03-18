#!/bin/bash
#v#!/usr/bin/env Rscript

#PBS -N bloom_45
#PBS -l nodes=1:ppn=1,walltime=99:00:00
#PBS -l mem=20gb
#PBS -q hydro
#PBS -e bloom_45_error.txt
#PBS -o bloom_45_output.txt
#PBS -M h.noorazar@yahoo.com
#PBS -m abe
cd $PBS_O_WORKDIR

# Ensure a clean running environment:
module purge

# Load modules (if needed)
module load R/R-3.2.2_gcc

/home/hnoorazar/cleaner_codes/drivers/bloom_driver.R rcp45

exit 0
