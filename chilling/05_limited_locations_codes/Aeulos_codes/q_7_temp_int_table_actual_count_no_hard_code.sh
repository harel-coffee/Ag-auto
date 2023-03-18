#!/bin/bash

#PBS -V
#PBS -N table

#PBS -l nodes=1:ppn=1,walltime=2:00:00
#PBS -l mem=10gb
#PBS -q hydro

#PBS -k o
##PBS -j oe
#PBS -e /home/hnoorazar/chilling_codes/hourly_limited_locations/error/table.E
#PBS -o /home/hnoorazar/chilling_codes/hourly_limited_locations/error/table.O
#PBS -m abe

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
# module load R

Rscript --vanilla /home/hnoorazar/chilling_codes/hourly_limited_locations/03_7_temp_int_table_actual_count_no_hard_code.R

echo
echo "----- DONE -----"
echo

exit 0
