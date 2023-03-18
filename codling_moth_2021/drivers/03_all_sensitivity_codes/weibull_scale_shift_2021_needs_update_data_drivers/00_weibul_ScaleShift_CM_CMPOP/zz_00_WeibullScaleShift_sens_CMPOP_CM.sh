#!/bin/bash
##v#!/usr/bin/env Rscript # commented out on Feb 18, 2021

# job name
#PBS -N sens_1_loc_orig_hist
#PBS -l nodes=1:ppn=1,walltime=06:00:00
#PBS -l mem=10gb
##PBS -q hydro
#PBS -e /home/hnoorazar/codling_moth_2021/drivers/03_all_sensitivity_codes/00_weibul_ScaleShift_CM_CMPOP/error/sens_1_orig_hist_e
#PBS -o /home/hnoorazar/codling_moth_2021/drivers/03_all_sensitivity_codes/00_weibul_ScaleShift_CM_CMPOP/error/sens_1_orig_hist_o
#PBS -M h.noorazar@yahoo.com
#PBS -m abe

######################################################################## 2018 begins

# cd $PBS_O_WORKDIR
# # First we ensure a clean running environment:
# module purge
# # Load modules (if needed)
# module load R/R-3.2.2_gcc



# ./00_WeibullScaleShift_sens_CMPOP_CM.R
# exit 0

######################################################################## 2018 ends
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


cd /home/hnoorazar/codling_moth_2021/drivers/03_all_sensitivity_codes/00_weibul_ScaleShift_CM_CMPOP
Rscript --vanilla ./00_WeibullScaleShift_sens_CMPOP_CM.R

echo
echo "----- DONE -----"
echo

exit 0


