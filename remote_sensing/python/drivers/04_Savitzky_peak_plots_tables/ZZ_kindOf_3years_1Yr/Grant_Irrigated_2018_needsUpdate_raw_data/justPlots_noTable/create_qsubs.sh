#!/bin/bash
cd /home/hnoorazar/remote_sensing_codes/02_Savitzky_my_peak/00_peak_tables_and_plots/Grant_Irrigated_2018_Savitzky/

outer=1
for indeks in EVI NDVI
  do
  for do_plot in 0 1
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
          sed -i s/do_plot/"$do_plot"/g    ./qsubs/q_$outer.sh
          let "outer+=1" 
        done
      done
    done
  done  
done