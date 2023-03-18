#!/bin/bash

## Export all environment variables 
## in the qsub command's environment to the
## batch job.
#PBS -V

## Define a job name
#PBS -N 02_obs_bloom_cuts

## Define compute options
#PBS -l nodes=1:dev:ppn=1
#PBS -l mem=10gb
#PBS -l walltime=6:00:00
##PBS -q hydro

## Define path for output & error logs
#PBS -k o
  ##PBS -j oe
#PBS -e /home/hnoorazar/bloom_codes/02_bloomCut_first_frost/error/obs_bloom.e
#PBS -o /home/hnoorazar/bloom_codes/02_bloomCut_first_frost/error/obs_bloom.o

## Define path for reporting
#PBS -m abe

echo
echo We are now in $PWD.
echo

cd /data/hydro/users/Hossein/bloom/01_binary_to_bloom/observed/

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
module load gcc/7.3.0
module load r/3.5.1/gcc/7.3.0
module load r/3.5.1

Rscript --vanilla /home/hnoorazar/bloom_codes/02_bloomCut_first_frost/02_d_observed_bloom.R

echo
echo "----- DONE -----"
echo

exit 0
