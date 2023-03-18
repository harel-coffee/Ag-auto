#!/bin/bash
cd /home/hnoorazar/NASA/02_remove_outliers_n_jumps/01_train_remove_jumps_JFD/

outer=1

for county in AdamBenton2016 FranklinYakima2018 Grant2017 Monterey2014 Walla2015
do
  for indeks in EVI NDVI
  do
    cp template_JFD.sh ./qsubs/q_JFD$outer.sh
    sed -i s/outer/"$outer"/g    ./qsubs/q_JFD$outer.sh
    sed -i s/indeks/"$indeks"/g  ./qsubs/q_JFD$outer.sh
    sed -i s/county/"$county"/g  ./qsubs/q_JFD$outer.sh
    let "outer+=1" 
  done
done