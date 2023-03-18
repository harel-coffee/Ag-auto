#!/bin/bash
cd /home/hnoorazar/NASA/05_SOS_detection_tables/

outer=1

for county in AdamBenton2016 FranklinYakima2018 Grant2017 Monterey2014
do
  for indeks in EVI NDVI
  do
    for SEOS_cut in 3 4 5
    do
      cp 03_train_table_template_JFD.sh ./qsubs/q_train_JFD_$outer.sh
      sed -i s/outer/"$outer"/g         ./qsubs/q_train_JFD_$outer.sh
      sed -i s/county/"$county"/g       ./qsubs/q_train_JFD_$outer.sh
      sed -i s/indeks/"$indeks"/g       ./qsubs/q_train_JFD_$outer.sh
      sed -i s/SEOS_cut/"$SEOS_cut"/g   ./qsubs/q_train_JFD_$outer.sh
      let "outer+=1" 
    done
  done
done
