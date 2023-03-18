#!/bin/bash
cd /home/hnoorazar/remote_sensing_codes/00_Kirti_Mike_initial_plots_Grant/01_acr_per_cult_table

outer=1
for county in "Grant"
do 
  for year in 2016 2017
  do 
    for freedom_df in 5 6 7 8 9
    do
      for look_ahead in 8 9 10 11 12
      do
        cp templated_dbl_peak_acr.sh ./qsubs/q_$outer.sh
        sed -i s/outer/"$outer"/g            ./qsubs/q_$outer.sh
        sed -i s/county/"$county"/g          ./qsubs/q_$outer.sh
        sed -i s/year/"$year"/g              ./qsubs/q_$outer.sh
        sed -i s/freedom_df/"$freedom_df"/g  ./qsubs/q_$outer.sh
        sed -i s/look_ahead/"$look_ahead"/g  ./qsubs/q_$outer.sh
        let "outer+=1" 
      done
     done  
  done
done