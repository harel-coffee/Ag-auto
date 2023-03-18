#!/bin/bash

#PBS -V
#PBS -N precip_type_Can_emission_type_int_file
#PBS -l mem=5gb

##PBS -l nodes=1:ppn=1,walltime=6:00:00
##PBS -q batch

##PBS -l nodes=1:ppn=1,walltime=99:00:00
##PBS -q hydro

#PBS -l nodes=1:ppn=1,walltime=2:00:00
#PBS -q fast

#PBS -k o
#PBS -e /home/hnoorazar/analog_codes/00_post_biofix/02_find_analogs_county_avg/error/precip_type_Can_emission_type_int_file.e
#PBS -o /home/hnoorazar/analog_codes/00_post_biofix/02_find_analogs_county_avg/error/precip_type_Can_emission_type_int_file.o
#PBS -m abe

cd /home/hnoorazar/analog_codes/00_post_biofix/02_find_analogs_county_avg

echo
echo We are in the $PWD directory
echo

module purge

# Load R
module load udunits/2.2.20
module load libxml2/2.9.4
module load gdal/2.1.2_gcc proj/4.9.2
module load gcc/7.3.0 r/3.5.1/gcc/7.3.0

Rscript --vanilla d_analog_cnty_avg_biofixed.R precip_type CanESM2 emission_type int_file

echo
echo "----- DONE -----"
echo

exit 0


