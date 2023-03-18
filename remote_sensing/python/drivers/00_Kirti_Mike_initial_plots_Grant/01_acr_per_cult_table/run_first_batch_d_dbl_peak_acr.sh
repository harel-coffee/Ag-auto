#!/bin/bash

cd /home/hnoorazar/remote_sensing_codes/00_Kirti_Mike_initial_plots_Grant/01_acr_per_cult_table/qsubs/
for runname in {1..25}
do
qsub ./q_$runname.sh
done
