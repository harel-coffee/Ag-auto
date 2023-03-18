#!/bin/bash

#PBS -V
#PBS -N 01_mod
#PBS -l mem=2gb

#PBS -l nodes=1:ppn=1,walltime=6:00:00
###PBS -q fast
#PBS -t 1-72

#PBS -k o
##PBS -j oe
#PBS -e /home/hnoorazar/lagoon_codes/01_rain_snow/01_rain_portions/error/mod_E
#PBS -o /home/hnoorazar/lagoon_codes/01_rain_snow/01_rain_portions/error/mod_O
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
# module load R

# new job for each directory index, up to max arrayid
cd ${dir_list[((${PBS_ARRAYID} - 1))]}

Rscript --vanilla /home/hnoorazar/lagoon_codes/01_rain_snow/01_rain_portions/00_d_modeled.R

echo
echo "----- DONE -----"
echo

exit 0
