#!/bin/bash

cd /home/hnoorazar/remote_sensing_codes/04_Savitzky_peak_plots_tables/00_peak_tables_and_plots_Aug26/02_2Yrs_regular_table_AllCY/qsubs/
for runname in {1..180}
do
qsub ./q_$runname.sh
done
