#!/bin/bash
cd /home/hnoorazar/NASA/02_remove_outliers_n_jumps/01_Grant_remove_jumps_JFD/

outer=1

for indeks in EVI NDVI
do
  cp template_JFD.sh ./qsubs/q_Grant_JFD$outer.sh
  sed -i s/outer/"$outer"/g    ./qsubs/q_Grant_JFD$outer.sh
  sed -i s/indeks/"$indeks"/g  ./qsubs/q_Grant_JFD$outer.sh
  let "outer+=1" 
done