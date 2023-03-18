#!/bin/bash
cd /home/h.noorazar/NASA/Kamiak_ML_Oct17

outer=1
for indeks in EVI NDVI
do
  for smooth_type in SG regular
  do
    for SR_ratio in 3 4 5 6 7 8
    do
      cp KNN_Oct17_accuracyScoring.sh       ./qsubs/KNN_Oct17_accuracyScoring$outer.sh
      sed -i s/outer/"$outer"/g             ./qsubs/KNN_Oct17_accuracyScoring$outer.sh
      sed -i s/indeks/"$indeks"/g           ./qsubs/KNN_Oct17_accuracyScoring$outer.sh
      sed -i s/smooth_type/"$smooth_type"/g ./qsubs/KNN_Oct17_accuracyScoring$outer.sh
      sed -i s/SR_ratio/"$SR_ratio"/g       ./qsubs/KNN_Oct17_accuracyScoring$outer.sh
      let "outer+=1" 
    done
  done
done