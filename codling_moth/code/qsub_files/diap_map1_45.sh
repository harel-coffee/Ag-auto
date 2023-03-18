#!/bin/bash

#PBS -N diap_map1_45
#PBS -l nodes=1:ppn=1,walltime=99:00:00
#PBS -l mem=20gb
#PBS -q hydro
#PBS -e /home/hnoorazar/cleaner_codes/qsub_files/error/E_diap_map1_45
#PBS -o /home/hnoorazar/cleaner_codes/qsub_files/error/o.diap_map1_45
#PBS -m abe

cd $PBS_O_WORKDIR

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

Rscript --vanilla /home/hnoorazar/cleaner_codes/drivers/diapause_map1.R rcp45

echo
echo "----- DONE -----"
echo

exit 0
