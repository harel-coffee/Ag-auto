#!/bin/bash
cd /home/h.noorazar/NASA/trend/clean_plots_4_DL

outer=1
for indeks in EVI NDVI
do
  for smooth_type in SG regular
  do
    batch_no=1
    while [ $batch_no -le 40 ]
    do
      cp trend_DLPlots_temp.sh              ./qsubs/trend_DLPlot_$indeks$smooth_type$batch_no.sh
      sed -i s/indeks/"$indeks"/g           ./qsubs/trend_DLPlot_$indeks$smooth_type$batch_no.sh
      sed -i s/smooth_type/"$smooth_type"/g ./qsubs/trend_DLPlot_$indeks$smooth_type$batch_no.sh
      sed -i s/batch_no/"$batch_no"/g       ./qsubs/trend_DLPlot_$indeks$smooth_type$batch_no.sh
      let "outer+=1"
      let "batch_no+=1"
    done
  done
done