#!/bin/bash
cd /home/hnoorazar/NASA/04_SG

outer=1
for random_or_all in random all
do
  for indeks in EVI NDVI
  do
    cp 01_SG_Grant_template_JFD.sh ./qsubs/q_Grant_JFD$outer.sh
    sed -i s/outer/"$outer"/g    ./qsubs/q_Grant_JFD$outer.sh
    sed -i s/indeks/"$indeks"/g  ./qsubs/q_Grant_JFD$outer.sh
    sed -i s/random_or_all/"$random_or_all"/g  ./qsubs/q_Grant_JFD$outer.sh
    let "outer+=1" 
  done
done