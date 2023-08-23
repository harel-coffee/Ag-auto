#!/bin/bash

cd /home/h.noorazar/NASA/trend/clean_plots_4_DL/qsubs

#for runname in {1..160}
# do

for indeks in EVI NDVI
do
  for smooth_type in SG regular
  do
    batch_no=1
    while [ $batch_no -le 40 ]
    do
    sbatch ./trend_DLPlot_pre2008_$indeks$smooth_type$batch_no.sh
    let "batch_no+=1"
    done
  done
done
