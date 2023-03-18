#!/bin/bash
cd /home/hnoorazar/remote_sensing_codes/00_Kirti_Mike_initial_plots_Grant/00_peak_tables_and_plots/

outer=1
for freedom_df in 4 5 6 7 8 9
do
  for look_ahead in 5 6 7 # 8 9 10 11 12
  do
    cp template.sh ./qsubs/q_$outer.sh
    sed -i s/outer/"$outer"/g          ./qsubs/q_$outer.sh
    sed -i s/freedom_df/"$freedom_df"/g  ./qsubs/q_$outer.sh
    sed -i s/look_ahead/"$look_ahead"/g    ./qsubs/q_$outer.sh
    let "outer+=1" 
  done
done  
