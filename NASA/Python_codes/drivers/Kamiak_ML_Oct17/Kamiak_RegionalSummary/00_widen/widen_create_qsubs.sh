#!/bin/bash
cd /home/h.noorazar/NASA/regionalStat/00_widen

outer=1
for indeks in EVI NDVI
do
  for smooth_type in SG regular
  do
    cp widenTemp.sh                       ./qsubs/widenTemp$outer.sh
    sed -i s/outer/"$outer"/g             ./qsubs/widenTemp$outer.sh
    sed -i s/indeks/"$indeks"/g           ./qsubs/widenTemp$outer.sh
    sed -i s/smooth_type/"$smooth_type"/g ./qsubs/widenTemp$outer.sh
    let "outer+=1"
  done
done