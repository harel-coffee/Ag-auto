#!/bin/bash

cd /home/hnoorazar/NASA/05_SOS_detection_plots/qsubs/
for runname in {1..3}
do
qsub ./q_Grant_$runname.sh
done
