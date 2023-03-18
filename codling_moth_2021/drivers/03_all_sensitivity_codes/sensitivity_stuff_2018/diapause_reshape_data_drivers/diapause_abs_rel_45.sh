#!/bin/bash
#v#!/usr/bin/env Rscript

#PBS -N diap_abs_rel_45
#PBS -l nodes=1:ppn=1,walltime=99:00:00
#PBS -l mem=2gb
#PBS -q hydro
#PBS -e diap_abs_rel_45_e.txt
#PBS -o diap_abs_rel_45_o.txt
#PBS -m abe
cd $PBS_O_WORKDIR
# Ensure a clean running environment:
module purge

# Load modules (if needed)
module load R/R-3.2.2_gcc
./diap_new_funcs.R rcp45
exit 0
