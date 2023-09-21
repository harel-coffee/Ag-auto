#!/bin/bash
cd /home/h.noorazar/NASA/trend/01_train_DeskReject/qsubs

for ML_model in DL
do
  for indeks in NDVI
  do
    for smooth in SG
    do
      for trainID in 1 2 3 4 5 6
      do
        for SR in 3 4 5 6 7 8
        do
          sbatch ./train_$indeks$smooth$ML_model$trainID$SR.sh
        done
      done
    done
  done
done
