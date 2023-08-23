#!/bin/bash
cd /home/hnoorazar/NASA/03_regularize_fillGap

outer=1

for indeks in EVI NDVI
do
  batchNo=1
  while [ $batchNo -le 40 ]
  do
    cp 02_reg_inters_template_JFD_pre2008.sh ./qsubs/q_inters_JFD_pre2008_$outer.sh
    sed -i s/outer/"$outer"/g                ./qsubs/q_inters_JFD_pre2008_$outer.sh
    sed -i s/indeks/"$indeks"/g              ./qsubs/q_inters_JFD_pre2008_$outer.sh
    sed -i s/batchNo/"$batchNo"/g            ./qsubs/q_inters_JFD_pre2008_$outer.sh
    let "batchNo+=1" 
    let "outer+=1" 
  done
done