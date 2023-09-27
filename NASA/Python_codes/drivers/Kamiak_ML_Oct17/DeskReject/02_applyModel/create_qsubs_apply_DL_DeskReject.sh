#!/bin/bash
cd /home/h.noorazar/NASA/DeskReject/02_apply_DeskReject

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
          cp temp_apply_DL_DeskReject.sh ./qsubs/apply_$indeks$smooth$ML_model$trainID$SR.sh
          sed -i s/indeks/"$indeks"/g    ./qsubs/apply_$indeks$smooth$ML_model$trainID$SR.sh
          sed -i s/smooth/"$smooth"/g    ./qsubs/apply_$indeks$smooth$ML_model$trainID$SR.sh
          sed -i s/trainID/"$trainID"/g  ./qsubs/apply_$indeks$smooth$ML_model$trainID$SR.sh
          sed -i s/SR/"$SR"/g            ./qsubs/apply_$indeks$smooth$ML_model$trainID$SR.sh
          let "outer+=1"
        done
      done
    done
  done
done