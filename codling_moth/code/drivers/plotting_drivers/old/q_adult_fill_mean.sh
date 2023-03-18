#!/bin/bash

#PBS -V

#PBS -N adult_fill_mu
#PBS -l nodes=1:ppn=1,walltime=01:00:00
#PBS -l mem=40gb
#PBS -q hydro

#PBS -k o
  ##PBS -j oe

#PBS -e /home/hnoorazar/cleaner_codes/drivers/error/adult_fill_mu.e
#PBS -o /home/hnoorazar/cleaner_codes/drivers/error/adult_fill_mu.o

cd /home/hnoorazar/cleaner_codes/drivers

echo
echo We are now in $PWD.
echo

module purge

# Load R
module load udunits/2.2.20
module load libxml2/2.9.4
module load gdal/2.1.2_gcc proj/4.9.2
module load gcc/7.3.0 r/3.5.1/gcc/7.3.0

./d_adult_DoY_fill_mean.R rcp45

exit 0

