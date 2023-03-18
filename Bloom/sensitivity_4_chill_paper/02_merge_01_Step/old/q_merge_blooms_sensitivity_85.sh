#!/bin/bash

#PBS -V
#PBS -N merge_blooms_85

#PBS -l nodes=1:ppn=1,walltime=6:00:00
#PBS -l mem=30gb
##PBS -q hydro

#PBS -k o
#PBS -e /home/hnoorazar/bloom_codes/sensitivity_4_chill_paper/02_merge_01_Step/error/merge_blooms_85.e
#PBS -o /home/hnoorazar/bloom_codes/sensitivity_4_chill_paper/02_merge_01_Step/error/merge_blooms_85.o
#PBS -m abe

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

Rscript --vanilla /home/hnoorazar/bloom_codes/sensitivity_4_chill_paper/02_merge_01_Step/02_d_merge_blooms_sensitivity_RCP85.R

echo
echo "----- DONE -----"
echo

exit 0
