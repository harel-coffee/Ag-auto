#!/bin/bash
#v#!/usr/bin/env Rscript

#PBS -N generations_85
#PBS -l nodes=1:ppn=1,walltime=99:00:00
#PBS -l mem=256mb
#PBS -q hydro
#PBS -e generations_85_e.txt
#PBS -o generations_85_o.txt
#PBS -m abe
cd $PBS_O_WORKDIR

# Ensure a clean running environment:
module purge

# Load modules (if needed)
module load R/R-3.2.2_gcc

/home/hnoorazar/cleaner_codes/drivers/sensitivity_driver/sensitivity_generations.R rcp85

exit 0
