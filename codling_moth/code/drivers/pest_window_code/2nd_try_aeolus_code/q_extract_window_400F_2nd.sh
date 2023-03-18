#!/bin/bash
#!/usr/bin/env Rscript

#PBS -N extract_window_400F_2nd
#PBS -l nodes=1:ppn=1,walltime=99:00:00
#PBS -l mem=5gb
#PBS -q hydro
#PBS -e E_extract_window_400F_2nd.txt
#PBS -o O_extract_window_400F_2nd.txt
#PBS -m abe
cd $PBS_O_WORKDIR
# Ensure a clean running environment:
module purge
# Load modules (if needed)
module load R/R-3.2.2_gcc 

./d_extract_window_400F_2nd.R

exit 0
