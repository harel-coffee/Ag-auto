#!/bin/bash
cd /home/hnoorazar/remote_sensing_codes/02_Savitzky_my_peak/00_Eastern_tables_and_plots/00_2Yrs_notRegular_Grant_2017/plots

outer=1

for jumps in yes no
do
  for indeks in EVI NDVI
  do
    for irrigated_only in 0 1
    do
      for SF_year in 2017
      do
        cp template.sh ./qsubs/q_$outer.sh
        sed -i s/outer/"$outer"/g          ./qsubs/q_$outer.sh
        sed -i s/jumps/"$jumps"/g          ./qsubs/q_$outer.sh
        sed -i s/indeks/"$indeks"/g          ./qsubs/q_$outer.sh
        sed -i s/irrigated_only/"$irrigated_only"/g   ./qsubs/q_$outer.sh
        sed -i s/SF_year/"$SF_year"/g  ./qsubs/q_$outer.sh
        let "outer+=1" 
      done
    done  
  done
done