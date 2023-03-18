#!/bin/bash
#v#!/usr/bin/env Rscript

#PBS -N ext_quan_2_85
#PBS -l nodes=1:ppn=1,walltime=99:00:00
#PBS -l mem=20gb
#PBS -q hydro
#PBS -e E_ext_quan_sep.txt
#PBS -o O_ext_quan_sep.txt
#PBS -m abe
cd $PBS_O_WORKDIR
# Ensure a clean running environment:
module purge
# Load modules (if needed)
module load R/R-3.2.2_gcc

./d_extract_quantile_sep_2.R 85.rds

exit 0
