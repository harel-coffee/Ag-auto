#!/bin/bash
cd /home/hnoorazar/NASA/02_remove_outliers_n_jumps/01_intersect_remove_jumps_JFD/

outer=1

for indeks in EVI NDVI
do
  batch_number=1
  while [ $batch_number -le 40 ]
  do
    cp template_JFD_intersect.sh             ./qsubs/q_JFD$outer.sh
    sed -i s/outer/"$outer"/g                ./qsubs/q_JFD$outer.sh
    sed -i s/indeks/"$indeks"/g              ./qsubs/q_JFD$outer.sh
    sed -i s/batch_number/"$batch_number"/g  ./qsubs/q_JFD$outer.sh
    let "batch_number+=1" 
    let "outer+=1" 
  done
done
