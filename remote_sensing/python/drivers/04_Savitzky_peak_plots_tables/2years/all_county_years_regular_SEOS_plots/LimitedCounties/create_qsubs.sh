#!/bin/bash
cd /home/hnoorazar/remote_sensing_codes/04_Savitzky_peak_plots_tables/00_peak_tables_and_plots_Aug26/01_2Yrs_regular_plots/limitedYC/

outer=1

for county in Grant 'Walla_Walla' Whitman Asotin Garfield Ferry Franklin Columbia Adams Benton Chelan Douglas Kittitas Klickitat Lincoln Okanogan Spokane Stevens Yakima 'Pend_Oreille'
do
  for irrigated_only in 0 1
  do
    for indeks in EVI
    do
      for SF_year in 2017
      do
        for jumps in no
        do 
          for SEOS_cut in 33 44 55
          do 
            cp template.sh ./qsubs/q_$outer.sh
            sed -i s/outer/"$outer"/g     ./qsubs/q_$outer.sh
            sed -i s/jumps/"$jumps"/g     ./qsubs/q_$outer.sh
            sed -i s/county/"$county"/g   ./qsubs/q_$outer.sh
            sed -i s/indeks/"$indeks"/g   ./qsubs/q_$outer.sh
            sed -i s/SF_year/"$SF_year"/g ./qsubs/q_$outer.sh
            sed -i s/SEOS_cut/"$SEOS_cut"/g ./qsubs/q_$outer.sh
            sed -i s/irrigated_only/"$irrigated_only"/g   ./qsubs/q_$outer.sh
            let "outer+=1"
          done
        done
      done
    done  
  done
done