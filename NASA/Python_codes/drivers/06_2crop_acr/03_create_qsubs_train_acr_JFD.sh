#!/bin/bash
cd /home/hnoorazar/NASA/06_2crop_acr/

outer=1

for indeks in EVI NDVI
do  
  cp 03_train_acr_template_JFD.sh ./qsubs/q_train_JFD_$outer.sh
  sed -i s/outer/"$outer"/g         ./qsubs/q_train_JFD_$outer.sh
  sed -i s/indeks/"$indeks"/g       ./qsubs/q_train_JFD_$outer.sh
  let "outer+=1" 
done
