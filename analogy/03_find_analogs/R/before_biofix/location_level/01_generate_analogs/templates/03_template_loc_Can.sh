#!/bin/bash

#PBS -V
#PBS -N precip_type_gen_3_type_Can_emission_type_int_file
#PBS -l mem=5gb

##PBS -l nodes=1:ppn=1,walltime=6:00:00
##PBS -q batch

##PBS -l nodes=1:ppn=1,walltime=99:00:00
##PBS -q hydro

#PBS -l nodes=1:ppn=1,walltime=2:00:00
#PBS -q fast

#PBS -k o
  ##PBS -j oe
#PBS -e /home/hnoorazar/analog_codes/03_find_analogs/location_level/error/precip_type_gen_3_type_Can_emission_type_int_file.e.txt
#PBS -o /home/hnoorazar/analog_codes/03_find_analogs/location_level/error/precip_type_gen_3_type_Can_emission_type_int_file.o.txt
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

Rscript --vanilla /home/hnoorazar/analog_codes/03_find_analogs/location_level/d_analog_location.R precip_type gen_3_type CanESM2 emission_type int_file

echo
echo "----- DONE -----"
echo

exit 0


