#!/bin/bash

cd /home/h.noorazar/NASA/trend/01_trend_preds/qsubs

for ML_model in RF DL SVM KNN
do 
  for indeks in EVI NDVI
  do
    for smooth in SG regular
    do
      batch_no=1
      while [ $batch_no -le 40 ]
      do
        sbatch ./trend_ML_preds_$indeks$smooth$batch_no$ML_model.sh
        let "batch_no+=1"
      done
    done
  done
done
