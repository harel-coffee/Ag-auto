#!/bin/bash
   
cd /home/hnoorazar/remote_sensing_codes/02_Savitzky_my_peak/00_Eastern_tables_and_plots/01_1Yr_regular_Grant_2017/plots/qsubs/
for runname in {1..4}
do
qsub ./q_$runname.sh
done
