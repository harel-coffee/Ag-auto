#!/bin/bash
cd /home/hnoorazar/NASA/02_remove_outliers_n_jumps/01_Grant_remove_jumps

outer=1

for indeks in EVI NDVI
do
  cp template.sh ./qsubs/q_Grant$outer.sh
  sed -i s/outer/"$outer"/g    ./qsubs/q_Grant$outer.sh
  sed -i s/indeks/"$indeks"/g  ./qsubs/q_Grant$outer.sh
  let "outer+=1" 
done