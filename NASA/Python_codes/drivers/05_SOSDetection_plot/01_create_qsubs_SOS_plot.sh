#!/bin/bash
cd /home/hnoorazar/NASA/05_SOS_detection_plots/

outer=1

for randCount in 100 500 all
do
  cp 01_SOS_plot_template.sh ./qsubs/q_Grant_$outer.sh
  sed -i s/outer/"$outer"/g          ./qsubs/q_Grant_$outer.sh
  sed -i s/randCount/"$randCount"/g  ./qsubs/q_Grant_$outer.sh
  let "outer+=1" 
done
