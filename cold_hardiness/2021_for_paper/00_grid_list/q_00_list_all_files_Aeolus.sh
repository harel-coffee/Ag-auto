#!/bin/bash

## Export all environment variables in the qsub command's environment to the
## batch job.
#PBS -V

## Define a job name
#PBS -N extract_all_grids

## Define compute options
#PBS -l nodes=1:dev:ppn=1
#PBS -l mem=1gb
#PBS -l walltime=01:00:00
##PBS -q hydro

## Define path for output & error logs
#PBS -k o
  ##PBS -j oe
#PBS -e /home/hnoorazar/hardiness_codes/00_grid_list/error/extract_all_grids_E
#PBS -o /home/hnoorazar/hardiness_codes/00_grid_list/error/extract_all_grids_O

## Define path for reporting
#PBS -m abe

echo
echo We are in the $PWD directory
echo

cd /data/hydro/jennylabcommon2/metdata/historical/UI_historical/VIC_Binary_CONUS_to_2016

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

Rscript --vanilla /home/hnoorazar/hardiness_codes/00_grid_list/00_list_all_files_Aeolus.R

echo
echo "----- DONE -----"
echo

exit 0
