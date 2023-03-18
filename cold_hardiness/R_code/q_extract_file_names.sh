#!/bin/bash

#PBS -V

## Define a job name
#PBS -N extract_file_names

## Define compute options
#PBS -l nodes=1:ppn=1,walltime=01:00:00
#PBS -l mem=2gb
#PBS -q fast

## Define path for output & error logs
#PBS -k o
##PBS -j oe
#PBS -e /home/hnoorazar/hardiness_codes/error/extract_file_names.e
#PBS -o /home/hnoorazar/hardiness_codes/error/extract_file_names.o

## Define path for reporting
#PBS -m abe

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
# module load R

echo
echo We are in the $PWD directory
echo

cd /data/hydro/users/kraghavendra/hardiness/output_data/Plots/facet/observed/

echo
echo We are now in $PWD.
echo

Rscript --vanilla /home/hnoorazar/hardiness_codes/extract_file_names.R

echo
echo "----- DONE -----"
echo

exit 0
