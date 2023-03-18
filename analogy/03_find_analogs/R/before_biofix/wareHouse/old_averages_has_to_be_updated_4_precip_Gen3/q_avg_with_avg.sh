#!/bin/bash

## Export all environment variables in the qsub command's environment to the
## batch job.
#PBS -V

## Define a job name
#PBS -N avg_avg_analog

## Define compute options
#PBS -l nodes=1:dev:ppn=1
#PBS -l mem=4gb
#PBS -l walltime=00:30:00
#PBS -q hydro

#PBS -k o
  ##PBS -j oe
#PBS -e /home/hnoorazar/analog_codes/03_find_analogs/error/m_avg_avg.e
#PBS -o /home/hnoorazar/analog_codes/03_find_analogs/error/m_avg_avg.o

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


Rscript --vanilla /home/hnoorazar/analog_codes/03_find_analogs/d_avg_with_avg.R

echo
echo "----- DONE -----"
echo

exit 0
