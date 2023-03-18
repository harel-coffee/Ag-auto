#!/bin/bash

#PBS -V
#PBS -N fill_median

#PBS -l nodes=1:ppn=1,walltime=02:00:00
#PBS -l mem=40gb
#PBS -q hydro
#PBS -k o
#PBS -e /home/hnoorazar/cleaner_codes/drivers/error/fill_median_E.txt
#PBS -o /home/hnoorazar/cleaner_codes/drivers/error/fill_median_O.txt

## Define path for reporting
#PBS -m abe

echo
echo We are in the $PWD directory
echo

cd $PBS_O_WORKDIR

echo
echo We are now in $PWD.
echo

module purge

# Load R
module load udunits/2.2.20
module load libxml2/2.9.4
module load gdal/2.1.2_gcc proj/4.9.2
module load gcc/7.3.0 r/3.5.1/gcc/7.3.0
module load gcc/7.3.0
module load r/3.5.1/gcc/7.3.0
module load r/3.5.1


Rscript --vanilla /home/hnoorazar/cleaner_codes/drivers/d_adult_fill_median_DoY.R 

echo
echo "----- DONE -----"
echo

exit 0
