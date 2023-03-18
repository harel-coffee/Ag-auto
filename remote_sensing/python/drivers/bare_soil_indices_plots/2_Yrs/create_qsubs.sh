#!/bin/bash
cd /home/hnoorazar/remote_sensing_codes/bare_soil_indices_plots

outer=1
for sf_year in 2017
do
  for county in Grant
  do
    for irrigated_only in 0 1
    do
      cp template.sh ./qsubs/q_$outer.sh
      sed -i s/outer/"$outer"/g      ./qsubs/q_$outer.sh
      sed -i s/county/"$county"/g    ./qsubs/q_$outer.sh
      sed -i s/sf_year/"$sf_year"/g  ./qsubs/q_$outer.sh
      sed -i s/irrigated_only/"$irrigated_only"/g  ./qsubs/q_$outer.sh
      let "outer+=1"
    done
  done
done  
