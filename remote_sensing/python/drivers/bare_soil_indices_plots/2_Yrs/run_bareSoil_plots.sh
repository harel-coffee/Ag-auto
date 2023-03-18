#!/bin/bash

cd /home/hnoorazar/remote_sensing_codes/bare_soil_indices_plots/qsubs/
for runname in {1..48}
do
qsub ./q_$runname.sh
done
