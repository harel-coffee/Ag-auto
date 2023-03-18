#!/bin/bash

#PBS -V
#PBS -N plot_bloom_fill_85_new
#PBS -l nodes=1:ppn=1,walltime=99:00:00
#PBS -l mem=40gb
#PBS -q hydro
#PBS -e /home/hnoorazar/cleaner_codes/qsub_files/error/E_plot_blm_fill_85_new.txt
#PBS -o /home/hnoorazar/cleaner_codes/qsub_files/error/O_plot_blm_fill_85_new.txt
#PBS -m abe
cd $PBS_O_WORKDIR

# Ensure a clean running environment:
module purge

# Load modules (if needed)
module load udunits/2.2.20
module load libxml2/2.9.4
module load gdal/2.1.2_gcc proj/4.9.2
module load gcc/7.3.0 r/3.5.1/gcc/7.3.0
module load gcc/7.3.0
module load r/3.5.1/gcc/7.3.0
module load r/3.5.1

Rscript --vanilla /home/hnoorazar/cleaner_codes/drivers/d_plot_bloom_new_filling.R rcp85

exit 0
