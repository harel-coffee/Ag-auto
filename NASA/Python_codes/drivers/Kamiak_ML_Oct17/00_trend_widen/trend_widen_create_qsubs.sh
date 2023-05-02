#!/bin/bash
cd /home/h.noorazar/NASA/trend/clean_plots_4_DL

outer=1
for ML_model in RF DL SVM kNN
do
  for indeks in EVI NDVI
  do
    for smooth in SG regular
    do
      batch_no=1
      while [ $batch_no -le 40 ]
      do
        cp trend_ML_preds_temp.sh        ./qsubs/trend_ML_preds_$indeks$smooth$batch_no$ML_model.sh
        sed -i s/indeks/"$indeks"/g      ./qsubs/trend_ML_preds_$indeks$smooth$batch_no$ML_model.sh
        sed -i s/smooth/"$smooth"/g      ./qsubs/trend_ML_preds_$indeks$smooth$batch_no$ML_model.sh
        sed -i s/batch_no/"$batch_no"/g  ./qsubs/trend_ML_preds_$indeks$smooth$batch_no$ML_model.sh
        sed -i s/ML_model/"$ML_model"/g  ./qsubs/trend_ML_preds_$indeks$smooth$batch_no$ML_model.sh
        let "outer+=1"
        let "batch_no+=1"
      done
    done
  done
done