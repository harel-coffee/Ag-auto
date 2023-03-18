#!/bin/bash

#PBS -V
#PBS -N read_mod_binary_for_frost

#PBS -l nodes=1:ppn=1,walltime=6:00:00
#PBS -l mem=2gb
#PBS -t 1-72

#PBS -k o
##PBS -j oe
#PBS -e /home/hnoorazar/chilling_codes/current_draft/03_frost_bloom/error/read_mod_binary_for_frost.e
#PBS -o /home/hnoorazar/chilling_codes/current_draft/03_frost_bloom/error/read_mod_binary_for_frost.o

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
done < <(find /data/hydro/jennylabcommon2/metdata/maca_v2_vic_binary/ -mindepth 2 -maxdepth 2 -type d -print0)

#echo
#echo "${dir_list[@]}"
#echo

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


# new job for each directory index, up to max arrayid
cd ${dir_list[((${PBS_ARRAYID} - 1))]}

Rscript --vanilla /home/hnoorazar/chilling_codes/current_draft/03_frost_bloom/d_read_binary_for_frost_modeled.R

echo
echo "----- DONE -----"
echo


exit 0
