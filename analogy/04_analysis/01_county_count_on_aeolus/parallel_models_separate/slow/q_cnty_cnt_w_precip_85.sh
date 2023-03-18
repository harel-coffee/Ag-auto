#!/bin/bash

## Export all environment variables in the qsub command's environment to the
## batch job.
#PBS -V

#PBS -N precip_85_all_model_names

#PBS -l nodes=1:ppn=1,walltime=99:00:00
#PBS -l mem=10gb
#PBS -q hydro

#PBS -k o
##PBS -j oe
#PBS -e /home/hnoorazar/analog_codes/04_analysis/parallel/error/E_precip_85_all_model_names
#PBS -o /home/hnoorazar/analog_codes/04_analysis/parallel/error/O_precip_85_all_model_names

#PBS -m abe

echo
echo We are in the $PWD directory
echo

cd /home/hnoorazar/analog_codes/04_analysis/parallel

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

Rscript --vanilla ./d_county_count_parallel.R rcp85 w_precip 2 all_model_names

echo
echo "----- DONE -----"
echo

exit 0
