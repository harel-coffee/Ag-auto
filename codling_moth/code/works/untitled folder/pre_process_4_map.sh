#!/bin/bash
#v#!/usr/bin/env Rscript

#PBS -N pre_process_4_map
#PBS -l nodes=1:ppn=1,walltime=99:00:00
#PBS -l mem=40gb
#PBS -q hydro
#PBS -e pre_process_4_map_error.txt
#PBS -o pre_process_4_map_output.txt
#PBS -m abe
cd $PBS_O_WORKDIR

# Ensure a clean running environment:
module purge

# Load modules (if needed)
module load R/R-3.2.2_gcc

/home/hnoorazar/cleaner_codes/drivers/pre_process_for_map.R rcp45

exit 0
