#!/bin/bash
#v#!/usr/bin/env Rscript
# job name

#PBS -N USA_H_CM
#PBS -l nodes=1:ppn=1,walltime=99:00:00
#PBS -l mem=10gb
#PBS -q hydro
#PBS -e /home/hnoorazar/cleaner_codes/drivers/All_USA/error/E_CM.txt
#PBS -o /home/hnoorazar/cleaner_codes/drivers/All_USA/error/O_CM.txt
#PBS -m abe
cd $PBS_O_WORKDIR

echo We are now in $PWD.

# First we ensure a clean running environment:
module purge

# Load R
module load udunits/2.2.20
module load libxml2/2.9.4
module load gdal/2.1.2_gcc proj/4.9.2
module load gcc/7.3.0 r/3.5.1/gcc/7.3.0

Rscript --vanilla ./d_USA_H_CM.R historical

echo
echo "----- DONE -----"
echo

exit 0
