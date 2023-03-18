#!/bin/bash

## Export all environment variables in the qsub command's environment to the
## batch job.
#PBS -V

## Define a job name
#PBS -N mod_utah_over

## Define compute options
#PBS -l nodes=1:dev:ppn=1
#PBS -l mem=10gb
#PBS -l walltime=10:00:00
#PBS -q hydro
#PBS -t 1-72

#PBS -k o
  ##PBS -j oe
#PBS -e /home/hnoorazar/chilling_codes/current_draft/02/utah_over_error/m_over.e
#PBS -o /home/hnoorazar/chilling_codes/current_draft/02/utah_over_error/m_over.o

## Define path for reporting
#PBS -m abe

echo
echo We are in the $PWD directory
echo

cd $PBS_O_WORKDIR

echo
echo We are now in $PWD.
echo

dir_list=()
while IFS= read -d $'\0' -r file ; do
dir_list=("${dir_list[@]}" "$file")

done < <(find /data/hydro/users/Hossein/chill/data_by_core/utah_model/01/modeled -mindepth 2 -maxdepth 2 -type d -print0)

echo
echo "${dir_list[@]}"
echo

# First we ensure a clean running environment:
module purge

# Load R
module load udunits/2.2.20
module load libxml2/2.9.4
module load gdal/2.1.2_gcc proj/4.9.2
module load gcc/7.3.0 r/3.5.1/gcc/7.3.0

# new job for each directory index, up to max arrayid
cd ${dir_list[((${PBS_ARRAYID} - 1))]}

Rscript --vanilla /home/hnoorazar/chilling_codes/current_draft/02/d_modeled.R "utah" "overlap"

echo
echo "----- DONE -----"
echo

exit 0
