#!/bin/bash
cd /home/h.noorazar/NASA/DeskReject/02_apply_DeskReject/qsubs

for ML_model in SVM ###RF KNN
do
  for indeks in NDVI
  do
    for smooth in SG
    do
      for trainID in 1 2 3 4 5 6
      do
        for SR in 3 4 5 6 7 8
        do
          sbatch ./apply_$indeks$smooth$ML_model$trainID$SR.sh
        done
      done
    done
  done
done
