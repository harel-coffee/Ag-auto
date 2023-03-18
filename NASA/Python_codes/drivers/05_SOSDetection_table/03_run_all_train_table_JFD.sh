#!/bin/bash

cd /home/hnoorazar/NASA/05_SOS_detection_tables/qsubs/
for runname in {1..24}
do
qsub ./q_train_JFD_$runname.sh
done
