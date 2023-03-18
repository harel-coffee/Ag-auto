#!/bin/bash

## Export all environment variables in the qsub command's environment to the
## batch job.
#PBS -V

## Define a job name
#PBS -N obs_dynamic_oct

## Define compute options
#PBS -l nodes=1:ppn=1,walltime=99:00:00
#PBS -l mem=20gb
#PBS -q hydro

## Define path for output & error logs
#PBS -k o
  ##PBS -j oe
#PBS -e /home/hnoorazar/cleaner_codes/qsub_files/error/obs_oct_dynamic.e
#PBS -o /home/hnoorazar/cleaner_codes/qsub_files/error/obs_oct_dynamic.o

## Define path for reporting
#PBS -m abe

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

Rscript --vanilla /home/hnoorazar/cleaner_codes/drivers/diapause_map1.R rcp85

echo
echo "----- DONE -----"
echo

exit 0


