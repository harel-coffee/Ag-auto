#!/bin/bash
#v#!/usr/bin/env Rscript

#PBS -N plot_adult_Aug
#PBS -l nodes=1:ppn=1,walltime=10:00:00
#PBS -l mem=20gb
#PBS -q hydro
#PBS -e plot_adult_Aug_e.txt
#PBS -o plot_adult_Aug_o.txt
#PBS -m abe
cd $PBS_O_WORKDIR

# Ensure a clean running environment:
module purge

# Load modules (if needed)
module load R/R-3.2.2_gcc

/home/hnoorazar/cleaner_codes/drivers/plot_adult_gen_aug23.R

exit 0
