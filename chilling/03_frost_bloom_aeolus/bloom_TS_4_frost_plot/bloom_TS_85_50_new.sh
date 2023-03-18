#!/bin/bash
#v#!/usr/bin/env Rscript

#PBS -N bloom_85_2015_0.5_new
#PBS -l nodes=1:ppn=1,walltime=6:00:00
#PBS -l mem=30gb
##PBS -q hydro
#PBS -e /home/hnoorazar/cleaner_codes/driver_bloom_TS_for_frost_plot/error/E_bloom_85_2015_0.5.txt
#PBS -o /home/hnoorazar/cleaner_codes/driver_bloom_TS_for_frost_plot/error/O_bloom_85_2015_0.5.txt
#PBS -m abe
cd $PBS_O_WORKDIR

# Ensure a clean running environment:
module purge

# Load modules (if needed)
module load R/R-3.2.2_gcc

/home/hnoorazar/cleaner_codes/driver_bloom_TS_for_frost_plot/bloom_TS.R rcp85 0.5

exit 0
