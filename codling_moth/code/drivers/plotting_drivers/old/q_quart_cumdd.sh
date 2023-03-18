#!/bin/bash
#v#!/usr/bin/env Rscript

#PBS -N plot_cumdd_quart_45
#PBS -l nodes=1:ppn=1,walltime=99:00:00
#PBS -l mem=20gb
#PBS -q hydro
#PBS -e plot_cumdd_quart_45_E.txt
#PBS -o plot_cumdd_quart_O.txt
#PBS -m abe
cd $PBS_O_WORKDIR
# Ensure a clean running environment:
module purge

# Load modules (if needed)
module load R/R-3.2.2_gcc
./d_quart_cumdd.R rcp45
exit 0
