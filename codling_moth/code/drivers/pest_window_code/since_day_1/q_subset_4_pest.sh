#!/bin/bash
#v#!/usr/bin/env Rscript

#PBS -N subset_4_pest
#PBS -l nodes=1:ppn=1,walltime=99:00:00
#PBS -l mem=12gb
#PBS -q hydro
#PBS -e E_subset_4_pest.txt
#PBS -o O_subset_4_pest.txt
#PBS -m abe
cd $PBS_O_WORKDIR
# Ensure a clean running environment:
module purge
# Load modules (if needed)
module load R/R-3.2.2_gcc

./subset_4_pest.R

exit 0
