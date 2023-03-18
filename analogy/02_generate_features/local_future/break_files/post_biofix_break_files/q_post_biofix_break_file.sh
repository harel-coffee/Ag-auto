#!/bin/bash

#PBS -V
#PBS -N break_files

#PBS -l nodes=1:ppn=1,walltime=2:00:00
#PBS -l mem=4gb
#PBS -q fast

#PBS -k o
#PBS -e /home/hnoorazar/analog_codes/00_post_biofix/01_break_files_loc_level/error/break_files.e
#PBS -o /home/hnoorazar/analog_codes/00_post_biofix/01_break_files_loc_level/error/break_files.o
#PBS -m abe

echo
echo We are in the $PWD directory
echo

module purge

module load udunits/2.2.20
module load libxml2/2.9.4
module load gdal/2.1.2_gcc proj/4.9.2
module load gcc/7.3.0 r/3.5.1/gcc/7.3.0

Rscript --vanilla /home/hnoorazar/analog_codes/00_post_biofix/01_break_files_loc_level/d_post_biofix_break_file.R

echo
echo "----- DONE -----"
echo

exit 0
