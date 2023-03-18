#!/bin/bash
cd /home/hnoorazar/remote_sensing_codes/05_confusionTables/00_create_tables

outer=1

for SEOS_cut in 33 44 55
do
  for indeks in EVI
  do
    for SF_year in 2016 2017 2018
    do
      for county in Grant Whitman Asotin Garfield Ferry Franklin Columbia Adams Benton Chelan Douglas Kittitas Klickitat Lincoln Okanogan Spokane Stevens Yakima 'Pend_Oreille' 'Walla_Walla'
      do
        cp template.sh                 ./qsubs/q_$outer.sh
        sed -i s/outer/"$outer"/g      ./qsubs/q_$outer.sh
        sed -i s/SEOS_cut/"$SEOS_cut"/g    ./qsubs/q_$outer.sh
        sed -i s/indeks/"$indeks"/g    ./qsubs/q_$outer.sh
        sed -i s/SF_year/"$SF_year"/g  ./qsubs/q_$outer.sh
        sed -i s/county/"$county"/g    ./qsubs/q_$outer.sh
        let "outer+=1" 
      done
    done
  done
done
