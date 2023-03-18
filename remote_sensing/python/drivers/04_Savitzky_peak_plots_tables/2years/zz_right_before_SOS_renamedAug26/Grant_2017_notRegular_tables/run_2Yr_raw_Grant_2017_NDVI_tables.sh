#!/bin/bash

cd /home/hnoorazar/remote_sensing_codes/02_Savitzky_my_peak/00_Eastern_tables_and_plots/00_2Yrs_raw_Grant_2017/tables/qsubs/
for runname in {49..96}
do
qsub ./q_$runname.sh
done
