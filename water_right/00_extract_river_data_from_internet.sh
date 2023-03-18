#!/bin/bash

#PBS -V
#PBS -N WA_rivers
#PBS -l mem=20gb

#PBS -l nodes=1:ppn=1,walltime=6:00:00
##PBS -q fast

#PBS -k o
##PBS -j oe
#PBS -e /home/hnoorazar/water_right_codes/error/WA_rivers_E
#PBS -o /home/hnoorazar/water_right_codes/error/WA_rivers_O
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
module load gcc/7.3.0
module load r/3.5.1/gcc/7.3.0
module load r/3.5.1
# module load R

# new job for each directory index, up to max arrayid
cd /home/hnoorazar/water_right_codes/
echo
echo We are now in $PWD.
echo

Rscript --vanilla ./extract_river_data_from_internet.R

echo
echo "----- DONE -----"
echo

exit 0
