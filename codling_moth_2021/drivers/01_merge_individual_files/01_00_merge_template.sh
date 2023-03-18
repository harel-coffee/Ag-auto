#!/bin/bash
##v#!/usr/bin/env Rscript # commented out on Feb 18, 2021

#PBS -N outer_merge_sub_dir_emission
#PBS -l nodes=1:ppn=1,walltime=06:00:00
#PBS -l mem=20gb
##PBS -q hydro
#PBS -e /home/hnoorazar/codling_moth_2021/drivers/01_merge_individual_files/error/sub_dir_emission_error.txt
#PBS -o /home/hnoorazar/codling_moth_2021/drivers/01_merge_individual_files/error/sub_dir_emission_output.txt
#PBS -m abe

######################## old start
# cd $PBS_O_WORKDIR

# Ensure a clean running environment:
# module purge

# Load modules (if needed)
# module load R/R-3.2.2_gcc


# /home/hnoorazar/codling_moth_2021/drivers/01_merge_individual_files/merge_driver.R sub_dir emission

######################## old end

echo
echo We are in $PWD.
echo

cd $PBS_O_WORKDIR

echo
echo We are in $PWD.
echo

# Ensure a clean running environment:
module purge

# Load modules (if needed)
# module load R/R-3.2.2_gcc

#
# New from chill
#
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

cd /home/hnoorazar/codling_moth_2021/drivers/01_merge_individual_files
Rscript --vanilla ./merge_driver.R sub_dir emission

echo
echo "----- DONE -----"
echo

exit 0


