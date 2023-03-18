#!/bin/bash
#v#!/usr/bin/env Rscript

#PBS -N d_sub_1000F
#PBS -l nodes=1:ppn=1,walltime=99:00:00
#PBS -l mem=20gb
#PBS -q hydro
#PBS -e E_sub_1000F.txt
#PBS -o O_sub_1000F.txt
#PBS -m abe
cd $PBS_O_WORKDIR
# Ensure a clean running environment:
module purge
# Load modules (if needed)
module load R/R-3.2.2_gcc

./d_subset_1000F.R

exit 0
