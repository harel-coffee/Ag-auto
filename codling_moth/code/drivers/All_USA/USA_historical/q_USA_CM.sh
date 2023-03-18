#!/bin/bash
#v#!/usr/bin/env Rscript
# job name

#PBS -N USA_H_CM
#PBS -l nodes=1:ppn=1,walltime=99:00:00
#PBS -l mem=10gb
#PBS -q hydro
#PBS -e /home/hnoorazar/cleaner_codes/drivers/All_USA/error/E_CM.txt
#PBS -o /home/hnoorazar/cleaner_codes/drivers/All_USA/error/O_CM.txt
#PBS -m abe
cd $PBS_O_WORKDIR

echo We are now in $PWD.

# First we ensure a clean running environment:
module purge
# Load modules (if needed)
module load R/R-3.2.2_gcc

./d_USA_H_CM.R historical

exit 0
