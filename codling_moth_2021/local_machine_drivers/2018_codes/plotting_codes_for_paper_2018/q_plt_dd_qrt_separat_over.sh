#!/bin/bash

#PBS -N plt_dd_qrt_separ_over
#PBS -l nodes=1:ppn=1,walltime=02:00:00
#PBS -l mem=20gb
#PBS -q hydro
#PBS -e /data/hydro/users/Hossein/codling_moth_new/local/processed/overlaping/cumdd_data/error/plot_e_sep.txt
#PBS -o /data/hydro/users/Hossein/codling_moth_new/local/processed/overlaping/cumdd_data/error/plot_o_sep.txt
#PBS -m abe

cd /data/hydro/users/Hossein/codling_moth_new/local/processed/overlaping/cumdd_data/

echo
echo We are in the $PWD directory
echo

# clean environment:
module purge

# Load R
module load udunits/2.2.20
module load libxml2/2.9.4
module load gdal/2.1.2_gcc proj/4.9.2
module load gcc/7.3.0 r/3.5.1/gcc/7.3.0

Rscript --vanilla ./plot_dd_separate_over.R

echo
echo "----- DONE -----"
echo

exit 0

