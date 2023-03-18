#!/bin/bash
##v#!/usr/bin/env Rscript # commented out on Feb 18, 2021

# job name
#PBS -N vertdd_driver

#PBS -l nodes=1:ppn=1,walltime=06:00:00
#PBS -l mem=10gb
##PBS -q hydro
#PBS -e /home/hnoorazar/codling_moth_2021/drivers/02_Diap_Gens_Vertdd/error/vertdd_error.txt
#PBS -o /home/hnoorazar/codling_moth_2021/drivers/02_Diap_Gens_Vertdd/error/vertdd_output.txt
#PBS -M h.noorazar@yahoo.com
#PBS -m abe


######################## old start
# cd $PBS_O_WORKDIR

# # First we ensure a clean running environment:
# module purge

# # Load modules (if needed)
# module load R/R-3.2.2_gcc

# /home/hnoorazar/codling_moth_2021/drivers/02_Diap_Gens_Vertdd/vertdd_driver.R
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


cd /home/hnoorazar/codling_moth_2021/drivers/02_Diap_Gens_Vertdd/ 
Rscript --vanilla ./vertdd_driver.R

echo
echo "----- DONE -----"
echo

exit 0


