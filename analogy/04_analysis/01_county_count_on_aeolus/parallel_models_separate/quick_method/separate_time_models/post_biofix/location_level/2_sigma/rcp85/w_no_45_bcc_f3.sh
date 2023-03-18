#!/bin/bash
#PBS -V
#PBS -N w_bcc_F3_85

#PBS -l nodes=1:ppn=1,walltime=6:00:00
#PBS -l mem=40gb
##PBS -q fast

#PBS -k o
##PBS -j oe
#PBS -e /home/hnoorazar/analog_codes/04_analysis/parallel/quick/error/E_w_bcc_F3_85
#PBS -o /home/hnoorazar/analog_codes/04_analysis/parallel/quick/error/O_w_bcc_F3_85
#PBS -m abe

echo
echo We are in the $PWD directory
echo

cd /home/hnoorazar/analog_codes/04_analysis/post_biofix/parallel/quick

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

Rscript --vanilla ./count_counties_quick.R rcp85 w_precip 2 bcc-csm1-1-m _2076_2095

echo
echo "----- DONE -----"
echo

exit 0
