#!/bin/bash
#PBS -V
#PBS -N detect_analogs_w_85_2

#PBS -l nodes=1:ppn=1,mem=40gb
#PBS -l walltime=06:00:00
#PBS -q batch

#PBS -k o
#PBS -e /home/hnoorazar/analog_codes/00_post_biofix/03_detect_analogs_4_plots/error/E_da_w_85_2
#PBS -o /home/hnoorazar/analog_codes/00_post_biofix/03_detect_analogs_4_plots/error/O_da_w_85_2
#PBS -m abe

cd /home/hnoorazar/analog_codes/00_post_biofix/03_detect_analogs_4_plots/01_find_intr_cnty_analogs
echo
echo We are now in $PWD.
echo

module purge
module load udunits/2.2.20
module load libxml2/2.9.4
module load gdal/2.1.2_gcc proj/4.9.2
module load gcc/7.3.0 r/3.5.1/gcc/7.3.0

Rscript --vanilla ./find_int_cnty_analogs.R w_precip rcp85 2

echo
echo "----- DONE -----"
echo

exit 0
