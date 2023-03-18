#!/bin/bash
#PBS -V
#PBS -N get_feat_hist

#PBS -l nodes=1:ppn=1,walltime=2:00:00
#PBS -l mem=40gb
#PBS -q fast

## Define path for output & error logs
#PBS -k o
#PBS -e /home/hnoorazar/analog_codes/00_biofix/error/get_feat_hist.e
#PBS -o /home/hnoorazar/analog_codes/00_biofix/error/get_feat_hist.o


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

Rscript --vanilla /home/hnoorazar/analog_codes/00_biofix/d_extract_hist_gdd_for_biofix.R

echo
echo "----- DONE -----"
echo

exit 0
