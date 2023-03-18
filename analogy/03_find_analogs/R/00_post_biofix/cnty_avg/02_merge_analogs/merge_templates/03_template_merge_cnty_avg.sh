#!/bin/bash

#PBS -V
#PBS -N merge_precip_type_emission_type_model_type

#PBS -l nodes=1:ppn=1,walltime=2:00:00
#PBS -l mem=40gb
#PBS -q fast

#PBS -k o
#PBS -e /home/hnoorazar/analog_codes/00_post_biofix/02_find_analogs_county_avg/merge/error/precip_type_emission_type_model_type.e
#PBS -o /home/hnoorazar/analog_codes/00_post_biofix/02_find_analogs_county_avg/merge/error/precip_type_emission_type_model_type.o
#PBS -m abe

cd /home/hnoorazar/analog_codes/00_post_biofix/02_find_analogs_county_avg/merge/
echo
echo We are in the $PWD directory
echo

module purge

# Load R
module load udunits/2.2.20
module load libxml2/2.9.4
module load gdal/2.1.2_gcc proj/4.9.2
module load gcc/7.3.0 r/3.5.1/gcc/7.3.0

Rscript --vanilla d_merge_analog_cnty_avg.R precip_type model_type emission_type

echo
echo "----- DONE -----"
echo

exit 0


