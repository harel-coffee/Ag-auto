#!/bin/bash
##v#!/usr/bin/env Rscript # commented out on Feb 18, 2021

# job name
#PBS -N LM_CM_emission_a_model_location
#PBS -l nodes=1:ppn=1,walltime=06:00:00
#PBS -l mem=20gb
##PBS -q hydro
#PBS -e /home/hnoorazar/codling_moth_2021/drivers/00_local_modeled/error/LM_CM_emission_a_model_location_error.txt
#PBS -o /home/hnoorazar/codling_moth_2021/drivers/00_local_modeled/error/LM_CM_emission_a_model_location_output.txt
#PBS -m abe


###################################################### Old, 2018 begin
# cd $PBS_O_WORKDIR

# # Ensure a clean running environment:
# module purge

# # Load modules (if needed)
# module load R/R-3.2.2_gcc

# /home/hnoorazar/codling_moth_2021/drivers/00_local_modeled/00_LM_CM_driver.R emission a_model location

###################################################### Old, 2018 end

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


cd /home/hnoorazar/codling_moth_2021/drivers/00_local_modeled/
Rscript --vanilla ./00_LM_CM_driver.R emission a_model location

echo
echo "----- DONE -----"
echo

exit 0
