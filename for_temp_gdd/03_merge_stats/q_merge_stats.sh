#!/bin/bash

## Export all environment variables in the qsub command's environment to the
## batch job.
#PBS -V

## Define a job name
#PBS -N merge_stats

## Define compute options
#PBS -l nodes=1:dev:ppn=1
#PBS -l mem=2gb
#PBS -l walltime=20:00:00
#PBS -q hydro

## Define path for output & error logs
#PBS -k o
  ##PBS -j oe
#PBS -e /data/hydro/users/Hossein/temp_gdd/error/merge_stats_E.txt
#PBS -o /data/hydro/users/Hossein/temp_gdd/error/merge_stats_O.txt

## Define path for reporting
#PBS -m abe

cd $PBS_O_WORKDIR

echo
echo We are now in $PWD.
echo

# First we ensure a clean running environment:
module purge

### ulimit -s 16384 # enlarge stack limit to 16 megs
### R --slave -e 'Cstack_info()["size"]'

# Load R
module load udunits/2.2.20
module load libxml2/2.9.4
module load gdal/2.1.2_gcc proj/4.9.2
module load gcc/7.3.0 r/3.5.1/gcc/7.3.0

Rscript --vanilla ./d_merge_statistics.R

echo
echo "----- DONE -----"
echo

exit 0
