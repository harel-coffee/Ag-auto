#!/bin/bash
cd /home/hnoorazar/NASA/02_remove_outliers_n_jumps/00_intersect_remove_outliers/

outer=1
for satellite in L4 L5_early L5_late L7
do
  for indeks in EVI NDVI
  do
    cp template_intersect_pre2008.sh  ./qsubs/q_pre2008_$outer.sh
    sed -i s/outer/"$outer"/g         ./qsubs/q_pre2008_$outer.sh
    sed -i s/indeks/"$indeks"/g       ./qsubs/q_pre2008_$outer.sh
    sed -i s/satellite/"$satellite"/g ./qsubs/q_pre2008_$outer.sh
    let "outer+=1" 
  done
done