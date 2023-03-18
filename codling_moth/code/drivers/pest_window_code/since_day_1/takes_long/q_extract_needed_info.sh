#!/bin/bash
#v#!/usr/bin/env Rscript

#PBS -N extract_info
#PBS -l nodes=1:ppn=1,walltime=99:00:00
#PBS -l mem=12gb
#PBS -q hydro
#PBS -e E_extract_info.txt
#PBS -o O_extract_info.txt
#PBS -m abe
cd $PBS_O_WORKDIR
# Ensure a clean running environment:
module purge
# Load modules (if needed)
module load R/R-3.2.2_gcc

./extract_needed_info.R

exit 0
