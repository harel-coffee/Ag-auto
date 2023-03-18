#!/bin/bash
#PBS -V
#PBS -N job_names

#PBS -l nodes=1:ppn=1,mem=4gb
#PBS -l walltime=02:00:00
#PBS -q batch

#PBS -k o
#PBS -e /home/hnoorazar/analog_codes/00_post_biofix/03_detect_analogs_4_plots/error/E_job_names
#PBS -o /home/hnoorazar/analog_codes/00_post_biofix/03_detect_analogs_4_plots/error/O_job_names
#PBS -m abe

echo
echo We are now in $PWD.
echo


module purge
module load udunits/2.2.20
module load libxml2/2.9.4
module load gdal/2.1.2_gcc proj/4.9.2
module load gcc/7.3.0 r/3.5.1/gcc/7.3.0

Rscript --vanilla /home/hnoorazar/analog_codes/00_post_biofix/03_detect_analogs_4_plots/d_list_of_job_names.R

echo
echo "----- DONE -----"
echo

exit 0
