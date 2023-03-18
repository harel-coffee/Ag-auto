#!/bin/bash
cd /home/hnoorazar/NASA/05_SOS_detection_plots/

outer=1

for county in AdamBenton2016 FranklinYakima2018 Grant2017 Walla2015
do
  cp 04_exactEval_plot_template_JFD.sh ./qsubs/q_exactEval_JFD_$outer.sh
  sed -i s/outer/"$outer"/g            ./qsubs/q_exactEval_JFD_$outer.sh
  sed -i s/county/"$county"/g          ./qsubs/q_exactEval_JFD_$outer.sh
  let "outer+=1" 
done
