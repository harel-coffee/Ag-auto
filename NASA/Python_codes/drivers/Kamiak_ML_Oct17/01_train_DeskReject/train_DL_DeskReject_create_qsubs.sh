#!/bin/bash
cd /home/h.noorazar/NASA/trend/01_train_DeskReject

outer=1
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
          cp train_DL_DeskReject_temp.sh ./qsubs/train_$indeks$smooth$ML_model$trainID$SR.sh
          sed -i s/indeks/"$indeks"/g    ./qsubs/train_$indeks$smooth$ML_model$trainID$SR.sh
          sed -i s/smooth/"$smooth"/g    ./qsubs/train_$indeks$smooth$ML_model$trainID$SR.sh
          sed -i s/trainID/"$trainID"/g  ./qsubs/train_$indeks$smooth$ML_model$trainID$SR.sh
          sed -i s/SR/"$SR"/g            ./qsubs/train_$indeks$smooth$ML_model$trainID$SR.sh
          let "outer+=1"
        done
      done
    done
  done
done