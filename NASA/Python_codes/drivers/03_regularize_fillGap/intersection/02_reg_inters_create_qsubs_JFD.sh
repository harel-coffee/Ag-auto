#!/bin/bash
cd /home/hnoorazar/NASA/03_regularize_fillGap

outer=1

for indeks in EVI NDVI
do
  batch=1
  while [ $batch -le 40 ]
  do
    cp 02_reg_inters_template_JFD.sh ./qsubs/q_inters_JFD$outer.sh
    sed -i s/outer/"$outer"/g        ./qsubs/q_inters_JFD$outer.sh
    sed -i s/indeks/"$indeks"/g      ./qsubs/q_inters_JFD$outer.sh
    sed -i s/batch/"$batch"/g        ./qsubs/q_inters_JFD$outer.sh
    let "batch+=1" 
    let "outer+=1" 
  done
done