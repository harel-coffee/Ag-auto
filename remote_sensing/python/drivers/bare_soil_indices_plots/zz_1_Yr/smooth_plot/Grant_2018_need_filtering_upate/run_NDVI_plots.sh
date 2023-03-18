#!/bin/bash

cd /home/hnoorazar/remote_sensing_codes/02_Savitzky_my_peak/00_peak_tables_and_plots/Grant_2018_smoothened_Plot_BareSoil/qsubs/
for runname in {49..96}
do
qsub ./q_$runname.sh
done
