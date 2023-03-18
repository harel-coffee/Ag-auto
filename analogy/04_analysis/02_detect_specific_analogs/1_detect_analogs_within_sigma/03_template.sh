#!/bin/bash
#PBS -V
#PBS -N emission_sigma_bd_int_name
#PBS -l mem=5gb

##PBS -l nodes=1:ppn=1,walltime=6:00:00
##PBS -q batch

##PBS -l nodes=1:ppn=1,walltime=99:00:00
##PBS -q hydro

#PBS -l nodes=1:ppn=1,walltime=2:00:00
#PBS -q fast

#PBS -k o
#PBS -e /home/hnoorazar/analog_codes/00_post_biofix/03_detect_analogs_4_plots/02_find_analogs_within_sigma/error/E_precip_emission_sigma_bd_int_name
#PBS -o /home/hnoorazar/analog_codes/00_post_biofix/03_detect_analogs_4_plots/02_find_analogs_within_sigma/error/O_precip_emission_sigma_bd_int_name
#PBS -m abe

cd /home/hnoorazar/analog_codes/00_post_biofix/03_detect_analogs_4_plots/02_find_analogs_within_sigma
echo
echo We are in the $PWD directory
echo

module purge

# Load R
module load udunits/2.2.20
module load libxml2/2.9.4
module load gdal/2.1.2_gcc proj/4.9.2
module load gcc/7.3.0 r/3.5.1/gcc/7.3.0

Rscript --vanilla  ./d_detect_within_sigma.R precip emission sigma_bd int_name

echo
echo "----- DONE -----"
echo

exit 0


