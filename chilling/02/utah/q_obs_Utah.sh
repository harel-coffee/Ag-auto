#!/bin/bash

## Export all environment variables in the qsub command's environment to the
## batch job.
#PBS -V

## Define a job name
#PBS -N obs_utah

## Define compute options
#PBS -l nodes=1:dev:ppn=1
#PBS -l mem=5gb
#PBS -l walltime=10:00:00
#PBS -q hydro

#PBS -k o
  ##PBS -j oe
#PBS -e /home/hnoorazar/chilling_codes/current_draft/02/observed_error/obs_utah.e
#PBS -o /home/hnoorazar/chilling_codes/current_draft/02/observed_error/obs_utah.o

## Define path for reporting
#PBS -m abe

echo
echo We are in the $PWD directory
echo

cd /data/hydro/users/Hossein/chill/data_by_core/utah_model/01/observed/

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
                  
Rscript --vanilla /home/hnoorazar/chilling_codes/current_draft/02/d_observed.R "utah"

echo
echo "----- DONE -----"
echo

exit 0
