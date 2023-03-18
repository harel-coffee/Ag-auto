#!/bin/bash

cd /home/hnoorazar/remote_sensing_codes/02_remove_outliers_n_jumps/2Yrs/02_merge_step_01/qsubs/
for runname in {1..6}
do
qsub ./q_$runname.sh
done
