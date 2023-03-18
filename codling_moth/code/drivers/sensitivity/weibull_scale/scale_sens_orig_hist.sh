#!/bin/bash
#v#!/usr/bin/env Rscript
# job name
#PBS -N sens_1_loc_orig_hist
#PBS -l nodes=1:ppn=1,walltime=70:00:00
#PBS -l mem=5gb
#PBS -q hydro
#PBS -e sens_1_orig_hist_e.txt
#PBS -o sens_1_orig_hist_o.txt
#PBS -M h.noorazar@yahoo.com
#PBS -m abe
cd $PBS_O_WORKDIR
# First we ensure a clean running environment:
module purge
# Load modules (if needed)
module load R/R-3.2.2_gcc
./scale_sens_orig_hist.R
exit 0
