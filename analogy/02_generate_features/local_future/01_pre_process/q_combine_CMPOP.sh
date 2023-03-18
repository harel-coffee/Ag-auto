#!/bin/bash
#PBS -V

#PBS -N combine_CMPOPS
#PBS -l nodes=1:dev:ppn=1
#PBS -l mem=30gb
#PBS -l walltime=5:00:00
#PBS -q hydro

## Define path for output & error logs
#PBS -k o
  ##PBS -j oe
#PBS -e /home/hnoorazar/analog_codes/02_generate_features/local/error/combine_CMPOPS.e
#PBS -o /home/hnoorazar/analog_codes/02_generate_features/local/error/combine_CMPOPS.o

## Define path for reporting
#PBS -m abe

echo
echo We are in the $PWD directory
echo

# First we ensure a clean running environment:
module purge

# Load R
module load udunits/2.2.20
module load libxml2/2.9.4
module load gdal/2.1.2_gcc proj/4.9.2
module load gcc/7.3.0 r/3.5.1/gcc/7.3.0

Rscript --vanilla /home/hnoorazar/analog_codes/02_generate_features/local/d_combine_CMPOPS.R

echo
echo "----- DONE -----"
echo

exit 0
