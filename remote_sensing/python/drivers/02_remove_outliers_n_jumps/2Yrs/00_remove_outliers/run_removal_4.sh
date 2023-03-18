#!/bin/bash

cd /home/hnoorazar/remote_sensing_codes/02_remove_outliers_n_jumps/2Yrs/00_remove_outliers/qsubs/
for runname in {561..641}
do
qsub ./q_$runname.sh
done
