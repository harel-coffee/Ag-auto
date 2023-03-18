#!/bin/bash
cd /home/hnoorazar/remote_sensing_codes/03_regularize_fillGap/01_regularize_2Yrs/

outer=1

for jumps in noJumps
do
  for indeks in EVI ## NDVI
  do
    for cloud_type in 70_cloud # 30_cloud 30_cloud_max 10_cloud 
    do
      for SF_year in 2016 2017 2018
      do
        for county in Grant Whitman Asotin Garfield Ferry Franklin Columbia Adams Benton Chelan Douglas Kittitas Klickitat Lincoln Okanogan Spokane Stevens Yakima 'Pend_Oreille' 'Walla_Walla'
        do
          cp 01_fillGaps_template.sh          ./fillGap_qsubs/q_$outer.sh
          sed -i s/outer/"$outer"/g           ./fillGap_qsubs/q_$outer.sh
          sed -i s/cloud_type/"$cloud_type"/g ./fillGap_qsubs/q_$outer.sh
          sed -i s/indeks/"$indeks"/g         ./fillGap_qsubs/q_$outer.sh
          sed -i s/jumps/"$jumps"/g           ./fillGap_qsubs/q_$outer.sh
          sed -i s/SF_year/"$SF_year"/g       ./fillGap_qsubs/q_$outer.sh
          sed -i s/county/"$county"/g         ./fillGap_qsubs/q_$outer.sh
          let "outer+=1" 
        done
      done  
    done
  done
done


