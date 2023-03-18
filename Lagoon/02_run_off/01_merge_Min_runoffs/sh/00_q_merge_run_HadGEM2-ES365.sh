#!/bin/bash

#PBS -V
#PBS -N merge_runoff_HadGEM2-ES365
#PBS -l mem=20gb

#PBS -l nodes=1:ppn=1,walltime=6:00:00

#PBS -k o
#PBS -e /home/hnoorazar/lagoon_codes/01_merge_runoff/error/merge_runoff_HadGEM2-ES365E
#PBS -o /home/hnoorazar/lagoon_codes/01_merge_runoff/error/merge_runoff_HadGEM2-ES365O
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

cd /home/hnoorazar/lagoon_codes/01_merge_runoff
Rscript --vanilla ./00_d_merge_runoff.R HadGEM2-ES365

echo
echo "----- DONE -----"
echo

exit 0
