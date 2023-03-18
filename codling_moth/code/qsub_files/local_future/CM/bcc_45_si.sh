#!/bin/bash
#v#!/usr/bin/env Rscript

#PBS -N bcc_45_s2
#PBS -l nodes=1:ppn=1,walltime=99:00:00
#PBS -l mem=5gb
#PBS -q hydro
#PBS -e bcc_45_s2_error.txt
#PBS -o bcc_45_s2_output.txt
#PBS -M h.noorazar@yahoo.com
#PBS -m abe
cd $PBS_O_WORKDIR

# Ensure a clean running environment:
module purge

# Load modules (if needed)
module load R/R-3.2.2_gcc

/home/hnoorazar/cleaner_codes/drivers/local_future/LF_CM_45_S2.R bcc-csm1-1-m

exit 0
