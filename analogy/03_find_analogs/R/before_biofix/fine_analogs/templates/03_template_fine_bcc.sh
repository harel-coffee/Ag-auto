#!/bin/bash

#PBS -V
#PBS -N precip_type_gen_3_type_bccm_emission_type_int_file

#PBS -l nodes=1:ppn=1,walltime=01:45:00
#PBS -l mem=4gb
#PBS -q hydro

#PBS -k o
  ##PBS -j oe
#PBS -e /home/hnoorazar/analog_codes/03_find_analogs/fine/error/precip_type_gen_3_type_bccm_emission_type_int_file.e
#PBS -o /home/hnoorazar/analog_codes/03_find_analogs/fine/error/precip_type_gen_3_type_bccm_emission_type_int_file.o
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

Rscript --vanilla /home/hnoorazar/analog_codes/03_find_analogs/fine/d_analog_fine.R precip_type gen_3_type bcc-csm1-1-m emission_type int_file

echo
echo "----- DONE -----"
echo

exit 0


