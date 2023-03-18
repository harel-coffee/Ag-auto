#!/bin/bash
cd /home/hnoorazar/remote_sensing_codes/04_Savitzky_peak_plots_tables/00_peak_tables_and_plots_Aug26/02_2Yrs_regular_table_AllCY

outer=1

for county in Grant Whitman Asotin Garfield Ferry Franklin Columbia Adams Benton Chelan Douglas Kittitas Klickitat Lincoln Okanogan Spokane Stevens Yakima 'Pend_Oreille' 'Walla_Walla'
do
  for SF_year in 2016 2017 2018
  do
    for indeks in EVI
    do
      for SG_params in 51 53 73 93
      do  
        for delt in .4
        do 
          for SEOS_cut in 33 44 55
          do
            cp template.sh ./qsubs/q_$outer.sh
            sed -i s/outer/"$outer"/g      ./qsubs/q_$outer.sh
            sed -i s/delt/"$delt"/g      ./qsubs/q_$outer.sh
            sed -i s/county/"$county"/g    ./qsubs/q_$outer.sh
            sed -i s/indeks/"$indeks"/g    ./qsubs/q_$outer.sh
            sed -i s/SF_year/"$SF_year"/g  ./qsubs/q_$outer.sh
            sed -i s/SEOS_cut/"$SEOS_cut"/g  ./qsubs/q_$outer.sh
            sed -i s/SG_params/"$SG_params"/g  ./qsubs/q_$outer.sh
            let "outer+=1" 
          done
        done
      done  
    done
  done
done