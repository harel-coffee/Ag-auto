#!/bin/bash
#!/usr/bin/env Rscript

#PBS -N ext_window_1000F
#PBS -l nodes=1:ppn=1,walltime=99:00:00
#PBS -l mem=5gb
#PBS -q hydro
#PBS -e E_ext_window_1000F.txt
#PBS -o O_ext_window_1000F.txt
#PBS -m abe
cd $PBS_O_WORKDIR
# Ensure a clean running environment:
module purge
# Load modules (if needed)
module load R/R-3.2.2_gcc 

./d_extract_window_1000F.R

exit 0
