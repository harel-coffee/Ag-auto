#!/bin/bash
#v#!/usr/bin/env Rscript

#PBS -N LA_50%_45
#PBS -l nodes=1:ppn=1,walltime=99:00:00
#PBS -l mem=20gb
#PBS -q hydro
#PBS -e E_LA_50%_45.txt
#PBS -o O_LA_50%_45.txt
#PBS -m abe
cd $PBS_O_WORKDIR
# Ensure a clean running environment:
module purge
# Load modules (if needed)
module load R/R-3.2.2_gcc

./d_extract_50.R 45.rds

exit 0
