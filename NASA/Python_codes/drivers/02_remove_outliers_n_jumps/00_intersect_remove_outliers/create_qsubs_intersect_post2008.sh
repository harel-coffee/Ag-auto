#!/bin/bash
cd /home/hnoorazar/NASA/02_remove_outliers_n_jumps/00_intersect_remove_outliers/

outer=1
# for satellite in L5 L7 L8
# do
for indeks in EVI NDVI
do
  cp template_intersect_post2008.sh ./qsubs/q_post2008_$outer.sh
  sed -i s/outer/"$outer"/g         ./qsubs/q_post2008_$outer.sh
  sed -i s/indeks/"$indeks"/g       ./qsubs/q_post2008_$outer.sh
  let "outer+=1" 
done
# done