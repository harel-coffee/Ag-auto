#!/bin/bash
#v#!/usr/bin/env Rscript

#PBS -N BNU_85_S2
#PBS -l nodes=1:ppn=1,walltime=99:00:00
#PBS -l mem=5gb
#PBS -q hydro
#PBS -e BNU_85_S2_error.txt
#PBS -o BNU_85_S2_output.txt
#PBS -M h.noorazar@yahoo.com
#PBS -m abe
cd $PBS_O_WORKDIR

# Ensure a clean running environment:
module purge

# Load modules (if needed)
module load R/R-3.2.2_gcc

/home/hnoorazar/cleaner_codes/drivers/local_future/LF_CM_85_S2.R BNU-ESM

exit 0
