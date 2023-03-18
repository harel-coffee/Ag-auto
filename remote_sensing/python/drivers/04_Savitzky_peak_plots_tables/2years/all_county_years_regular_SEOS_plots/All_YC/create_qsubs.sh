#!/bin/bash
cd /home/hnoorazar/remote_sensing_codes/04_Savitzky_peak_plots_tables/00_peak_tables_and_plots_Aug26/01_2Yrs_regular_plots/allYCF_plots/

outer=1

for SEOS_cut in 33 ## 44 55
do 
  for county in Grant Franklin Yakima 'Walla_Walla' Adams Benton Whitman Asotin Garfield Ferry Columbia Chelan Douglas Kittitas Klickitat Lincoln Okanogan Spokane Stevens 'Pend_Oreille'
  do
    for SF_year in 2017 2018 2016
    do
      for irrigated_only in 0 1
      do
        for indeks in EVI
        do
          for jumps in no
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