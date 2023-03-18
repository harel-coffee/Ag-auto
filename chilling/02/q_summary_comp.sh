#!/bin/bash

## Export all environment variables in the qsub command's environment to the
## batch job.
#PBS -V

## Define a job name
#PBS -N summary_comp

## Define compute options
#PBS -l nodes=1:dev:ppn=1, walltime=06:00:00
#PBS -l mem=10gb
#PBS -q hydro

## Define path for output & error logs
#PBS -k o
  ##PBS -j oe
#PBS -e /home/hnoorazar/chilling_codes/current_draft/02/error/summary_comp.e
#PBS -o /home/hnoorazar/chilling_codes/current_draft/02/error/summary_comp.o

## Define path for reporting
#PBS -m abe

echo
echo We are in the $PWD directory
echo

cd /home/hnoorazar/chilling_codes/current_draft/02/

echo
echo We are now in $PWD.
echo

# First we ensure a clean running environment:
module purge

# Load R
module load udunits/2.2.20
module load libxml2/2.9.4
module load gdal/2.1.2_gcc proj/4.9.2
module load gcc/7.3.0 r/3.5.1/gcc/7.3.0

Rscript --vanilla ./d_summary_comp.R

echo
echo "----- DONE -----"
echo

exit 0
