#!/bin/bash

## Export all environment variables in the 
## qsub command's environment to the
## batch job.
#PBS -V

## Define a job name
#PBS -N modeled_frost

## Define compute options
#PBS -l nodes=1:dev:ppn=1
#PBS -l mem=20gb
#PBS -l walltime=6:00:00
##PBS -q hydro
#PBS -t 1-60

## Define path for output & error logs
#PBS -k o
##PBS -j oe
#PBS -e /home/hnoorazar/bloom_codes/02_bloomCut_first_frost/error/modeled_frost_E.txt
#PBS -o /home/hnoorazar/bloom_codes/02_bloomCut_first_frost/error/modeled_frost_O.txt

## Define path for reporting
#PBS -m abe

cd $PBS_O_WORKDIR

echo
echo We are now in $PWD.
echo

dir_list=()
while IFS= read -d $'\0' -r file ; do
dir_list=("${dir_list[@]}" "$file")
done < <(find /data/hydro/users/Hossein/bloom/01_binary_to_bloom/modeled/ -mindepth 2 -maxdepth 2 -type d -print0)

# First we ensure a clean running environment:
module purge

# Load R
module load udunits/2.2.20
module load libxml2/2.9.4
module load gdal/2.1.2_gcc proj/4.9.2
module load gcc/7.3.0 r/3.5.1/gcc/7.3.0
module load gcc/7.3.0
module load r/3.5.1/gcc/7.3.0
module load r/3.5.1
# module load R

# new job for each directory index, up to max arrayid
cd ${dir_list[((${PBS_ARRAYID} - 1))]}

Rscript --vanilla /home/hnoorazar/bloom_codes/02_bloomCut_first_frost/02_d_modeled_frost.R

echo
echo "----- DONE -----"
echo

exit 0
