#!/bin/bash

cd /home/hnoorazar/NASA/05_SOS_detection_plots/qsubs/
for runname in {1..4}
do
qsub ./q_train_JFD_$runname.sh
done
