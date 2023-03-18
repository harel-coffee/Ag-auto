#!/bin/bash
#v#!/usr/bin/env Rscript

#PBS -N LM_CM
#PBS -l nodes=1:ppn=1,walltime=06:00:00
#PBS -l mem=20gb
##PBS -q hydro
#PBS -e /home/hnoorazar/codling_moth_2021/drivers/00_local_modeled/error/LM_CM_error.txt
#PBS -o /home/hnoorazar/codling_moth_2021/drivers/00_local_modeled/error/LM_CM_output.txt
#PBS -m abe
cd $PBS_O_WORKDIR

# Ensure a clean running environment:
module purge

# Load modules (if needed)
module load R/R-3.2.2_gcc

/home/hnoorazar/codling_moth_2021/drivers/00_local_modeled/00_LM_CM_driver.R

exit 0
