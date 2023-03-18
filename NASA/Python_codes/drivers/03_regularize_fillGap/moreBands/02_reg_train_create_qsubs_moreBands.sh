#!/bin/bash
cd /home/hnoorazar/NASA/03_regularize_fillGap

outer=1
for county in Walla2015 AdamBenton2016 Grant2017 FranklinYakima2018
do
  for indeks in blue green red NIR short_I_1 short_I_2
  do
    cp 02_reg_train_template_moreBands.sh ./qsubs/q_train_moreBands$outer.sh
    sed -i s/outer/"$outer"/g    ./qsubs/q_train_moreBands$outer.sh
    sed -i s/indeks/"$indeks"/g  ./qsubs/q_train_moreBands$outer.sh
    sed -i s/county/"$county"/g  ./qsubs/q_train_moreBands$outer.sh
    let "outer+=1" 
  done
done