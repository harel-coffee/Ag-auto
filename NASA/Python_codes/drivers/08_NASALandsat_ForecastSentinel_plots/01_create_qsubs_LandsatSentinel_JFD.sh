#!/bin/bash
cd /home/hnoorazar/NASA/08_NASALandsat_ForecastSentinel_plots

outer=1

for county in AdamBenton2016 FranklinYakima2018 Grant2017 Walla2015
do
  cp 01_LandsatSentinel_JFD_plot_template.sh ./qsubs/q_exactEval_JFD$outer.sh
  sed -i s/outer/"$outer"/g                  ./qsubs/q_exactEval_JFD$outer.sh
  sed -i s/county/"$county"/g                ./qsubs/q_exactEval_JFD$outer.sh
  let "outer+=1" 
done