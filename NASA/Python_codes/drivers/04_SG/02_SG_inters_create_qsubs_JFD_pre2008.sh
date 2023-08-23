#!/bin/bash
cd /home/hnoorazar/NASA/04_SG

outer=1
for indeks in EVI NDVI
do
  batch_size=1
  while [ $batch_size -le 40 ]
  do
    cp 02_SG_inters_template_JFD_pre2008.sh ./qsubs/q_inters_JFD_pre2008_$outer.sh
    sed -i s/outer/"$outer"/g               ./qsubs/q_inters_JFD_pre2008_$outer.sh
    sed -i s/indeks/"$indeks"/g             ./qsubs/q_inters_JFD_pre2008_$outer.sh
    sed -i s/batch_size/"$batch_size"/g     ./qsubs/q_inters_JFD_pre2008_$outer.sh
    let "outer+=1"
    let "batch_size+=1"
  done
done