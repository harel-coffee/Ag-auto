#!/bin/bash

## Export all environment variables in the qsub command's environment to the
## batch job.
#PBS -V

## Define a job name
#PBS -N subset

## Define compute options
#PBS -l nodes=1:ppn=1,walltime=4:00:00
#PBS -l mem=30gb
##PBS -q hydro

#PBS -k o
  ##PBS -j oe
#PBS -e /home/hnoorazar/chilling_codes/current_draft/03_frost_bloom/error/bloom_extract.e
#PBS -o /home/hnoorazar/chilling_codes/current_draft/03_frost_bloom/error/bloom_extract.o

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

Rscript --vanilla /home/hnoorazar/chilling_codes/current_draft/03_frost_bloom/extract_limited_cities_bloom.R

echo
echo "----- DONE -----"
echo

exit 0
