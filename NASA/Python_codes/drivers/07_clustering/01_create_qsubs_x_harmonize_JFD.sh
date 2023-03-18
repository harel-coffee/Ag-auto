#!/bin/bash
cd /home/hnoorazar/NASA/07_clustering/

outer=1

for indeks in EVI NDVI
do
  cp 01_x_harmonize_template_JFD.sh ./qsubs/q_train_JFD_harmonize_$outer.sh
  sed -i s/outer/"$outer"/g         ./qsubs/q_train_JFD_harmonize_$outer.sh
  sed -i s/indeks/"$indeks"/g       ./qsubs/q_train_JFD_harmonize_$outer.sh
  let "outer+=1"
done
