#!/bin/bash
cd /home/hnoorazar/remote_sensing_codes/02_Savitzky_my_peak/00_Eastern_tables_and_plots/00_2Yrs_raw_Grant_2017/tables

outer=1
for indeks in EVI NDVI
do
  for Sav_win_size in 3 5 7 9
  do
    for sav_order in 1 2 3
    do
      for delt in .1 .2 .3 .4
      do
        cp template.sh ./qsubs/q_$outer.sh
        sed -i s/outer/"$outer"/g          ./qsubs/q_$outer.sh
        sed -i s/indeks/"$indeks"/g          ./qsubs/q_$outer.sh
        sed -i s/Sav_win_size/"$Sav_win_size"/g          ./qsubs/q_$outer.sh
        sed -i s/sav_order/"$sav_order"/g  ./qsubs/q_$outer.sh
        sed -i s/delt/"$delt"/g      ./qsubs/q_$outer.sh
        let "outer+=1" 
      done
    done
  done  
done