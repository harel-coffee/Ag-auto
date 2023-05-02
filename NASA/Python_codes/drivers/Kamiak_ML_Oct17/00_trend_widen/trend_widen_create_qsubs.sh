#!/bin/bash
cd /home/h.noorazar/NASA/trend/00_trend_widen/

outer=1
for indeks in EVI NDVI
do
  for smooth in SG regular
  do
    batch_no=1
    while [ $batch_no -le 40 ]
    do
      cp trend_widen_temp.sh          ./qsubs/trend_widen_$indeks$smooth$batch_no.sh
      sed -i s/indeks/"$indeks"/g     ./qsubs/trend_widen_$indeks$smooth$batch_no.sh
      sed -i s/smooth/"$smooth"/g     ./qsubs/trend_widen_$indeks$smooth$batch_no.sh
      sed -i s/batch_no/"$batch_no"/g ./qsubs/trend_widen_$indeks$smooth$batch_no.sh
      let "outer+=1"
      let "batch_no+=1"
    done
  done
done
