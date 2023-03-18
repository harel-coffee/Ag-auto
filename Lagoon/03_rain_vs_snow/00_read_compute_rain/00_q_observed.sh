#!/bin/bash

#PBS -V
#PBS -N 01_obs
#PBS -l mem=2gb

#PBS -l nodes=1:ppn=1,walltime=6:00:00
###PBS -q fast

## Define path for output & error logs
#PBS -k o
#PBS -e /home/hnoorazar/lagoon_codes/01_rain_snow/01_rain_portions/error/obs_E
#PBS -o /home/hnoorazar/lagoon_codes/01_rain_snow/01_rain_portions/error/obs_O

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

Rscript --vanilla /home/hnoorazar/lagoon_codes/01_rain_snow/01_rain_portions/00_d_observed.R

echo
echo "----- DONE -----"
echo

exit 0
