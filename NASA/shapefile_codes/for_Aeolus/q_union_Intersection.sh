#!/bin/bash

#PBS -V
#PBS -N intersect_2017_2018

#PBS -l nodes=1:ppn=1,walltime=6:00:00
#PBS -l mem=40gb
##PBS -q fast

#PBS -k o
#PBS -e /home/hnoorazar/NASA/00_shapefileCodes/error/intersect_2017_2018.e
#PBS -o /home/hnoorazar/NASA/00_shapefileCodes/error/intersect_2017_2018.o
#PBS -m abe

cd /home/hnoorazar/NASA/00_shapefileCodes/
echo
echo We are in the $PWD directory
echo

module purge

# Load R
module load udunits/2.2.20
module load libxml2/2.9.4
module load gdal/2.3.2/gcc proj/4.9.2
module load gdal/2.3.2/gcc/7.3.0 # this is new. I do not know what proj/4.9.2 does
module load gcc/7.3.0 r/3.5.3/gcc/7.3.0
                      

Rscript --vanilla union_Intersection_2017_2018.R

echo
echo "----- DONE -----"
echo

exit 0


