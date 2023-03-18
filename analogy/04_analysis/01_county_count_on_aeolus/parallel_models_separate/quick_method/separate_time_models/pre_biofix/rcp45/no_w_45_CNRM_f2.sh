#!/bin/bash
#PBS -V
#PBS -N no_w_CNRM_F2_45

#PBS -l nodes=1:ppn=1,walltime=2:00:00
#PBS -l mem=40gb
#PBS -q fast

#PBS -k o
##PBS -j oe
#PBS -e /home/hnoorazar/analog_codes/04_analysis/parallel/quick/error/E_no_w_CNRM_F2_45
#PBS -o /home/hnoorazar/analog_codes/04_analysis/parallel/quick/error/O_no_w_CNRM_F2_45
#PBS -m abe

echo
echo We are in the $PWD directory
echo

cd /home/hnoorazar/analog_codes/04_analysis/parallel/quick

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

Rscript --vanilla ./no_no_count_counties_quick.R rcp45 no_precip 1 w_gen3 CNRM-CM5 _2051_2075

echo
echo "----- DONE -----"
echo

exit 0
