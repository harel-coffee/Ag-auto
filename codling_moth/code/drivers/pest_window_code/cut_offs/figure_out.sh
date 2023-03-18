#!/bin/bash
#v#!/usr/bin/env Rscript

#PBS -N figure
#PBS -l nodes=1:ppn=1,walltime=99:00:00
#PBS -l mem=20gb
#PBS -q hydro
#PBS -e E_figure.txt
#PBS -o O_figure.txt
#PBS -m abe
cd $PBS_O_WORKDIR
# Ensure a clean running environment:
module purge
# Load modules (if needed)
module load R/R-3.2.2_gcc 0.25

./figure_out.R

exit 0
