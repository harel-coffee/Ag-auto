#!/bin/bash
cd /home/hnoorazar/NASA/03_regularize_fillGap

outer=1
for county in AdamBenton2016 FranklinYakima2018 Grant2017 Monterey2014 Walla2015
do
  for indeks in EVI NDVI
  do
    cp 02_reg_train_template_JFD.sh ./qsubs/q_train_JFD$outer.sh
    sed -i s/outer/"$outer"/g    ./qsubs/q_train_JFD$outer.sh
    sed -i s/indeks/"$indeks"/g  ./qsubs/q_train_JFD$outer.sh
    sed -i s/county/"$county"/g  ./qsubs/q_train_JFD$outer.sh
    let "outer+=1" 
  done
done