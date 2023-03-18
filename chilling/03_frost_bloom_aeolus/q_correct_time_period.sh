#!/bin/bash

#PBS -V
#PBS -N f_correction

#PBS -l nodes=1:ppn=1,walltime=4:00:00
#PBS -l mem=30gb
#PBS -k o
#PBS -m abe

module purge
module load udunits/2.2.20
module load libxml2/2.9.4
module load gdal/2.1.2_gcc proj/4.9.2
module load gcc/7.3.0 r/3.5.1/gcc/7.3.0

Rscript --vanilla /home/hnoorazar/chilling_codes/current_draft/03_frost_bloom/correct_time_period.R

exit 0
