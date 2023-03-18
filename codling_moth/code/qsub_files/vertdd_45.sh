#!/bin/bash
#v#!/usr/bin/env Rscript

#PBS -N vertdd_45_2015
#PBS -l nodes=1:ppn=1,walltime=70:00:00
#PBS -l mem=20gb
#PBS -q hydro
#PBS -e /home/hnoorazar/cleaner_codes/qsub_files/error/E_vertdd_45_2015.txt
#PBS -o /home/hnoorazar/cleaner_codes/qsub_files/error/vertdd_45_2015_o.txt
#PBS -m abe
cd $PBS_O_WORKDIR

# Ensure a clean running environment:
module purge

# Load modules (if needed)
module load R/R-3.2.2_gcc

/home/hnoorazar/cleaner_codes/drivers/vertdd_driver.R combined_CMPOP_rcp45

exit 0
