#!/bin/bash

#PBS -V
#PBS -N observed_storm
#PBS -l mem=16gb

#PBS -l nodes=1:ppn=1,walltime=2:00:00
#PBS -q fast

#PBS -k o
#PBS -e /home/hnoorazar/lagoon_codes/01_cum_storm/storm/error/observed_storm_E
#PBS -o /home/hnoorazar/lagoon_codes/01_cum_storm/storm/error/observed_storm_O
#PBS -m abe

echo
echo We are in $PWD.
echo

module purge
module load udunits/2.2.20
module load libxml2/2.9.4
module load gdal/2.1.2_gcc proj/4.9.2
module load gcc/7.3.0 r/3.5.1/gcc/7.3.0
module load gcc/7.3.0
module load r/3.5.1/gcc/7.3.0
module load r/3.5.1

Rscript --vanilla /home/hnoorazar/lagoon_codes/01_cum_storm/storm/00_d_observed_storm.R

echo
echo "----- DONE -----"
echo

exit 0
