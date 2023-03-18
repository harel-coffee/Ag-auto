#!/bin/bash
#v#!/usr/bin/env Rscript
# job name
#PBS -N sens_15_bcc
#PBS -l nodes=1:ppn=1,walltime=70:00:00
#PBS -l mem=5gb
#PBS -q hydro
#PBS -e sens_15_bcc_e.txt
#PBS -o sens_15_bcc_o.txt
#PBS -M h.noorazar@yahoo.com
#PBS -m abe
cd $PBS_O_WORKDIR
# First we ensure a clean running environment:
module purge
# Load modules (if needed)
module load R/R-3.2.2_gcc
./sensitivity_shrunk_15.R bcc-csm1-1-m
exit 0