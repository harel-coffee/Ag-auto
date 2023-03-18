#!/bin/bash
cd /home/hnoorazar/NASA/02_remove_outliers_n_jumps/00_intersect_remove_outliers/

outer=1
for satellite in L5 L7 L8
do
  for indeks in EVI NDVI
  do
    cp template_intersect.sh          ./qsubs/q_$outer.sh
    sed -i s/outer/"$outer"/g         ./qsubs/q_$outer.sh
    sed -i s/indeks/"$indeks"/g       ./qsubs/q_$outer.sh
    sed -i s/satellite/"$satellite"/g ./qsubs/q_$outer.sh
    let "outer+=1" 
  done
done