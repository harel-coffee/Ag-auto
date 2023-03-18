#!/bin/bash
#v#!/usr/bin/env Rscript

#PBS -N merge_limited
#PBS -l nodes=1:ppn=1,walltime=10:00:00
#PBS -l mem=4gb
#PBS -q hydro
#PBS -e /home/hnoorazar/chilling_codes/hourly_limited_locations/error/merge_E.txt
#PBS -o /home/hnoorazar/chilling_codes/hourly_limited_locations/error/merge_O.txt
#PBS -m abe
cd /home/hnoorazar/chilling_codes/hourly_limited_locations

echo We are in the $PWD directory

# Ensure a clean running environment:
module purge

# Load modules (if needed)
module load R/R-3.2.2_gcc

./d_merge_intervals.R

exit 0
