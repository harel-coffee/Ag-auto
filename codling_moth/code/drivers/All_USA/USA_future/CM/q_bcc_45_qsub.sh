#!/bin/bash
#v#!/usr/bin/env Rscript

#PBS -N bcc-csm1-1-m
#PBS -l nodes=1:ppn=6,walltime=99:00:00
#PBS -l mem=10gb
#PBS -q hydro
#PBS -e E_bcc-csm1-1-m.txt
#PBS -o O.txt
#PBS -M h.noorazar@yahoo.com
#PBS -m abe
cd $PBS_O_WORKDIR

# Ensure a clean running environment:
module purge

# Load modules (if needed)
module load R/R-3.2.2_gcc

./A_all_us_future_45.R bcc-csm1-1-m

exit 0