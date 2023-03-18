#!/bin/bash
#v#!/usr/bin/env Rscript
# job name

#PBS -N USA_H_CMPOP
#PBS -l nodes=1:ppn=1,walltime=99:00:00
#PBS -l mem=10gb
#PBS -q hydro
#PBS -e E_CMPOP_85.txt
#PBS -o O.txt
#PBS -m abe
cd $PBS_O_WORKDIR
# First we ensure a clean running environment:
module purge
# Load modules (if needed)
module load R/R-3.2.2_gcc

./USA_F_CMPOP_85.R

exit 0
