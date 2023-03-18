#!/bin/bash

cd /home/hnoorazar/remote_sensing_codes/03_regularize_fillGap/01_regularize_2Yrs/qsubs/
for runname in {1..120}
do
qsub ./q_$runname.sh
done
