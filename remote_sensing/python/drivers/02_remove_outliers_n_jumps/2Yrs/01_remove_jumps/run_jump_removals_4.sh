#!/bin/bash

cd /home/hnoorazar/remote_sensing_codes/02_remove_outliers_n_jumps/2Yrs/01_remove_jumps/qsubs/
for runname in {541..640}
do
qsub ./q_$runname.sh
done
