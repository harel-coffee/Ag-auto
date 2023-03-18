#!/bin/bash

#PBS -V

## Define a job name
#PBS -N precip_type_gen_3_type_emission_type_time_type

#PBS -l nodes=1:ppn=1,walltime=20:00:00
#PBS -l mem=8gb
#PBS -q hydro

#PBS -k o
  ##PBS -j oe
#PBS -e /home/hnoorazar/analog_codes/04_analysis/error/precip_type_gen_3_type_emission_type_time_type.e
#PBS -o /home/hnoorazar/analog_codes/04_analysis/error/precip_type_gen_3_type_emission_type_time_type.o

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

Rscript --vanilla /home/hnoorazar/analog_codes/04_analysis/d_county_count.R precip_type gen_3_type emission_type time_type

echo
echo "----- DONE -----"
echo

exit 0
