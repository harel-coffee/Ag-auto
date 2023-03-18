#!/bin/bash
#v#!/usr/bin/env Rscript

#PBS -N LF_CMPOP_85
#PBS -l nodes=1:ppn=1,walltime=99:00:00
#PBS -l mem=5gb
#PBS -q hydro
#PBS -e LF_CMPOP_85_error.txt
#PBS -o LF_CMPOP_85_output.txt
#PBS -m abe
cd $PBS_O_WORKDIR

# Ensure a clean running environment:
module purge

# Load modules (if needed)
module load R/R-3.2.2_gcc

/home/hnoorazar/cleaner_codes/drivers/local_future/LF_CMPOP_85.R

exit 0
