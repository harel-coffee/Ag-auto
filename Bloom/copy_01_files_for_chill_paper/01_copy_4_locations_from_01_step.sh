#!/bin/bash

#PBS -V

## Define a job name
#PBS -N copy_01_for_chill

#PBS -l nodes=1:ppn=1,walltime=06:00:00
#PBS -l mem=25gb
##PBS -q hydro

#PBS -k o
  ##PBS -j oe
#PBS -e /home/hnoorazar/bloom_codes/copy_01_files_for_f_chill_paper/error/copy_01_for_chill.e
#PBS -o /home/hnoorazar/bloom_codes/copy_01_files_for_f_chill_paper/error/copy_01_for_chill.o
#PBS -m abe

echo
echo We are in the $PWD directory
echo

module purge

# Load R
module load udunits/2.2.20
module load libxml2/2.9.4
module load gdal/2.1.2_gcc proj/4.9.2
module load gcc/7.3.0 r/3.5.1/gcc/7.3.0

Rscript --vanilla /home/hnoorazar/bloom_codes/copy_01_files_for_f_chill_paper/01_copy_4_locations_from_01_step.R

echo
echo "----- DONE -----"
echo

exit 0


