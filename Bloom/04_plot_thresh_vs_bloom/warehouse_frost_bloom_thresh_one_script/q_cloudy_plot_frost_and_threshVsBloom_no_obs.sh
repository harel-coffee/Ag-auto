#!/bin/bash

#PBS -V
#PBS -N plot_FTB_no_obs

#PBS -l nodes=1:ppn=1,walltime=99:00:00
#PBS -l mem=30gb
#PBS -q hydro

#PBS -k o
#PBS -e /home/hnoorazar/bloom_codes/04_plot_thresh_vs_bloom/error/plot_FTB_no_obs.e
#PBS -o /home/hnoorazar/bloom_codes/04_plot_thresh_vs_bloom/error/plot_FTB_no_obs.o
#PBS -m abe

echo
echo We are now in $PWD.
echo

# First we ensure a clean running environment:
module purge

# Load R
module load udunits/2.2.20
module load libxml2/2.9.4
module load gdal/2.1.2_gcc proj/4.9.2
module load gcc/7.3.0 r/3.5.1/gcc/7.3.0
module load gcc/7.3.0
module load r/3.5.1/gcc/7.3.0
module load r/3.5.1

Rscript --vanilla /home/hnoorazar/bloom_codes/04_plot_thresh_vs_bloom/d_cloudy_plot_frost_and_threshVsBloom_no_obs.R

echo
echo "----- DONE -----"
echo

exit 0
