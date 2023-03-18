#!/bin/bash
#PBS -V
#PBS -N w_GG_F2_45

#PBS -l nodes=1:ppn=1,walltime=2:00:00
#PBS -l mem=40gb
#PBS -q fast

#PBS -k o
#PBS -e /home/hnoorazar/analog_codes/04_analysis/post_biofix/parallel/quick/cnty_avg/error/E_w_GG_F2_45
#PBS -o /home/hnoorazar/analog_codes/04_analysis/post_biofix/parallel/quick/cnty_avg/error/O_w_GG_F2_45
#PBS -m abe

echo
echo We are in the $PWD directory
echo

cd /home/hnoorazar/analog_codes/04_analysis/post_biofix/parallel/quick/cnty_avg

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

Rscript --vanilla ./count_counties_quick.R rcp45 w_precip 1 GFDL-ESM2G _2051_2075

echo
echo "----- DONE -----"
echo

exit 0
