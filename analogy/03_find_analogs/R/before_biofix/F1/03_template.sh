#!/bin/bash

#PBS -V

## Define a job name
#PBS -N F1_precip_type_gen_3_type_model_type_emission_type

#PBS -l nodes=1:ppn=1,walltime=99:00:00
#PBS -l mem=25gb
#PBS -q hydro

#PBS -k o
  ##PBS -j oe
#PBS -e /home/hnoorazar/analog_codes/03_find_analogs/error/precip_type_gen_3_type_model_type_emission_type.e
#PBS -o /home/hnoorazar/analog_codes/03_find_analogs/error/precip_type_gen_3_type_model_type_emission_type.o
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

Rscript --vanilla /home/hnoorazar/analog_codes/03_find_analogs/d_analog.R precip_type gen_3_type model_type emission_type

echo
echo "----- DONE -----"
echo

exit 0


