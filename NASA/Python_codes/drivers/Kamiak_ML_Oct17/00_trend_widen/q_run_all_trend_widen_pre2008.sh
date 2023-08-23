#!/bin/bash

cd /home/h.noorazar/NASA/trend/00_trend_widen/qsubs

for indeks in EVI NDVI
do
  for smooth in SG regular
  do
    batch_no=1
    while [ $batch_no -le 40 ]
    do
      sbatch ./trend_widen_pre2008_$indeks$smooth$batch_no.sh
      let "batch_no+=1"
    done
  done
done

