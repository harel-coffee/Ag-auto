#!/bin/bash
cd /home/hnoorazar/NASA/02_remove_outliers_n_jumps/00_Grant_remove_outliers/

outer=1

for random_or_all in random all
do
  for indeks in EVI NDVI
  do
    cp template.sh ./qsubs/q_$outer.sh
    sed -i s/outer/"$outer"/g    ./qsubs/q_$outer.sh
    sed -i s/indeks/"$indeks"/g  ./qsubs/q_$outer.sh
    sed -i s/random_or_all/"$random_or_all"/g  ./qsubs/q_$outer.sh
    let "outer+=1" 
  done
done