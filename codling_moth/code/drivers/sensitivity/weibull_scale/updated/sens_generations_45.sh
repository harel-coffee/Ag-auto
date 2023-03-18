#!/bin/bash
#v#!/usr/bin/env Rscript

#PBS -N generations_45
#PBS -l nodes=1:ppn=1,walltime=12:00:00
#PBS -l mem=256mb
#PBS -q hydro
#PBS -e generations_45_e.txt
#PBS -o generations_45_o.txt
#PBS -m abe
cd $PBS_O_WORKDIR

# Ensure a clean running environment:
module purge

# Load modules (if needed)
module load R/R-3.2.2_gcc

./sensitivity_generations.R rcp45

exit 0
