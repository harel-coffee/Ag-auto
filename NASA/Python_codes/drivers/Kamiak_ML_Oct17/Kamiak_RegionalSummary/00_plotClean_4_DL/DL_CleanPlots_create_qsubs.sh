#!/bin/bash
cd /home/h.noorazar/NASA/regionalStat/00_plotClean_4_DL

outer=1
for indeks in EVI NDVI
do
  for smooth_type in SG regular
  do
    cp DL_CleanPlots_Temp.sh              ./qsubs/DL_CleanPlots$outer.sh
    sed -i s/outer/"$outer"/g             ./qsubs/DL_CleanPlots$outer.sh
    sed -i s/indeks/"$indeks"/g           ./qsubs/DL_CleanPlots$outer.sh
    sed -i s/smooth_type/"$smooth_type"/g ./qsubs/DL_CleanPlots$outer.sh
    let "outer+=1"
  done
done