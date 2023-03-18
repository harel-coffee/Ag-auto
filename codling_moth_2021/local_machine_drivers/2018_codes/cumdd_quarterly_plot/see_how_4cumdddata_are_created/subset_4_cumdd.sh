#!/bin/bash
#v#!/usr/bin/env Rscript

#PBS -N subset_4_cumdd
#PBS -l nodes=1:ppn=1,walltime=99:00:00
#PBS -l mem=40gb
#PBS -q hydro
#PBS -e /home/hnoorazar/cleaner_codes/qsub_files/error/subset_4_cumdd_e.txt
#PBS -o /home/hnoorazar/cleaner_codes/qsub_files/error/subset_4_cumdd_o.txt
#PBS -m abe
cd $PBS_O_WORKDIR
# Ensure a clean running environment:
module purge
# Load modules (if needed)
module load R/R-3.2.2_gcc
/home/hnoorazar/cleaner_codes/drivers/subset_4_cumdd.R

exit 0
