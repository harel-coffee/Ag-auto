#!/bin/bash

## Define a job name
#PBS -N chunk_file_outer

#PBS -l nodes=1:ppn=1,walltime=6:00:00
#PBS -l mem=10gb
#PBS -e /home/hnoorazar/lagoon_codes/01_run_offs/01_cum_runs/qsubs/error/chunk_file_outer_E
#PBS -o /home/hnoorazar/lagoon_codes/01_run_offs/01_cum_runs/qsubs/error/chunk_file_outer_O
#PBS -m abe

echo
echo We are in $PWD.
echo

cd /home/hnoorazar/lagoon_codes/01_run_offs/01_cum_runs

module purge
module load udunits/2.2.20
module load libxml2/2.9.4
module load gdal/2.1.2_gcc proj/4.9.2
module load gcc/7.3.0 r/3.5.1/gcc/7.3.0
module load gcc/7.3.0
module load r/3.5.1/gcc/7.3.0
module load r/3.5.1

echo 
echo We are in the $PWD directory
echo 

Rscript --vanilla ./d_chunky_cum_run.R fileN
