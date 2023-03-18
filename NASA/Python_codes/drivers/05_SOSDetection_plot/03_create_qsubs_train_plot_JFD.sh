#!/bin/bash
cd /home/hnoorazar/NASA/05_SOS_detection_plots/

outer=1

for county in AdamBenton2016 FranklinYakima2018 Grant2017 Monterey2014 Walla2015
do
  cp 03_train_plot_template_JFD.sh ./qsubs/q_train_JFD_$outer.sh
  sed -i s/outer/"$outer"/g        ./qsubs/q_train_JFD_$outer.sh
  sed -i s/county/"$county"/g      ./qsubs/q_train_JFD_$outer.sh
  let "outer+=1" 
done
