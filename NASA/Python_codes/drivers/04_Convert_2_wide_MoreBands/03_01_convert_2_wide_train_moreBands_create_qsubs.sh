#!/bin/bash
cd /home/hnoorazar/NASA/03_01_widen_MoreBands

outer=1
for indeks in blue green red NIR short_I_1 short_I_2
do
  cp 03_01_convert_2_wide_train_moreBands_template.sh ./qsubs/q_train_moreBands_widen$outer.sh
  sed -i s/outer/"$outer"/g    ./qsubs/q_train_moreBands_widen$outer.sh
  sed -i s/indeks/"$indeks"/g  ./qsubs/q_train_moreBands_widen$outer.sh
  sed -i s/county/"$county"/g  ./qsubs/q_train_moreBands_widen$outer.sh
  let "outer+=1" 
done