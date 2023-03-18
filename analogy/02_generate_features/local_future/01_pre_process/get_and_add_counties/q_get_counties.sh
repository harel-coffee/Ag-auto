#!/bin/bash

#PBS -N get_counties
#PBS -l nodes=1:ppn=1,walltime=2:00:00
#PBS -l mem=20gb
#PBS -q hydro
#PBS -e /home/hnoorazar/analog_codes/02_generate_features/usa/get_and_add_counties/error/get_counties_e
#PBS -o /home/hnoorazar/analog_codes/02_generate_features/usa/get_and_add_counties/error/get_counties_o

#PBS -m abe
cd $PBS_O_WORKDIR

# Ensure a clean running environment:
module purge

# Load R
module load udunits/2.2.20
module load libxml2/2.9.4
module load gdal/2.1.2_gcc proj/4.9.2
module load gcc/7.3.0 r/3.5.1/gcc/7.3.0

Rscript --vanilla /home/hnoorazar/analog_codes/02_generate_features/usa/get_and_add_counties/d_get_counties.R

echo
echo "----- DONE -----"
echo

exit 0
