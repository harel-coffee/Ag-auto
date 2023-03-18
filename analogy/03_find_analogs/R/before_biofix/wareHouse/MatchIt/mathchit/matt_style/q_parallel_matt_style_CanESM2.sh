#!/bin/bash

## Export all environment variables in the qsub command's environment to the
## batch job.
#PBS -V

## Define a job name
#PBS -N match_85_Can

## Define compute options
#PBS -l nodes=1:dev:ppn=1
#PBS -l mem=5gb
#PBS -l walltime=60:00:00
#PBS -q hydro
#PBS -t 1-300

#PBS -k o
  ##PBS -j oe
#PBS -e /home/hnoorazar/analog_codes/03_find_analogs/R_codes/matchit/error/Can_m_non.e
#PBS -o /home/hnoorazar/analog_codes/03_find_analogs/R_codes/matchit/error/Can_m_non.o

## Define path for reporting
#PBS -m abe

echo
echo line 26 of bash We are in the $PWD directory
echo

# cd /data/hydro/users/Hossein/analog/local/ready_features/broken_down_location_level_coarse/rcp85/

echo
echo line 32 of bash We are now in $PWD.
echo

dir_list=()

while IFS="" read -d $'\0' -r file ; do
dir_list=("${dir_list[@]}" "$file")
done < <(find /data/hydro/users/Hossein/analog/local/ready_features/broken_down_location_level_coarse/rcp85/CanESM2/ -mindepth 1 -maxdepth 1 -type d -print0)

echo Line 42 of bash
echo "${dir_list[@]}"
echo Line 44 of bash

# First we ensure a clean running environment:
module purge

# Load R
module load udunits/2.2.20
module load libxml2/2.9.4
module load gdal/2.1.2_gcc proj/4.9.2
module load gcc/7.3.0 r/3.5.1/gcc/7.3.0

# new job for each directory index, up to max arrayid
cd ${dir_list[((${PBS_ARRAYID} - 1))]}

echo Line 59 of bash critical $PWD

Rscript --vanilla /home/hnoorazar/analog_codes/03_find_analogs/R_codes/matchit/d_matt_style.R

echo
echo "----- DONE -----"
echo

exit 0
