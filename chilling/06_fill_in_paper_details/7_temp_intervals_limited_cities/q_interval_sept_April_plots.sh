#!/bin/bash

## Export all environment variables in the qsub command's environment to the
## batch job.
#PBS -V

#PBS -N sept_April_specific_locs_separate_for_paper

#PBS -l nodes=1:dev:ppn=1,walltime=01:00:00
#PBS -l mem=100gb

#PBS -k o
  ##PBS -j oe
#PBS -e /home/hnoorazar/chilling_codes/current_draft/03_make_figures/7_intervals/error/sept_April_specific_locs_separate_for_paper.e
#PBS -o /home/hnoorazar/chilling_codes/current_draft/03_make_figures/7_intervals/error/sept_April_specific_locs_separate_for_paper.o

#PBS -m abe

cd /home/hnoorazar/chilling_codes/current_draft/03_make_figures/7_intervals/

echo
echo We are now in $PWD.
echo

# First we ensure a clean running environment:
module purge

# Load R
module load udunits/2.2.20
module load libxml2/2.9.4
module load gdal/2.1.2_gcc proj/4.9.2
module load gcc/7.3.0 r/3.5.1/gcc/7.3.0

Rscript --vanilla ./sept_April_specific_locs_separate_for_paper.R

echo
echo "----- DONE -----"
echo

exit 0
