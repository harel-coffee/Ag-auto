#!/bin/bash

## Export all environment variables in the qsub command's environment to the
## batch job.
#PBS -V

#PBS -N no_w_45

#PBS -l nodes=1:ppn=1,walltime=6:00:00
#PBS -l mem=10gb
##PBS -q hydro

#PBS -k o
##PBS -j oe
#PBS -e /home/hnoorazar/analog_codes/04_analysis/parallel/quick/error/E_no_precip_45
#PBS -o /home/hnoorazar/analog_codes/04_analysis/parallel/quick/error/O_no_precip_45

#PBS -m abe

echo
echo We are in the $PWD directory
echo

cd /home/hnoorazar/analog_codes/04_analysis/parallel/quick

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

Rscript --vanilla ./count_counties_quick.R rcp45 no_precip 1 w_gen3

echo
echo "----- DONE -----"
echo

exit 0
