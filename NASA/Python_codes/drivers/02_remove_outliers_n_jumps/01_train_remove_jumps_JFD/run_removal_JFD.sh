#!/bin/bash

cd /home/hnoorazar/NASA/02_remove_outliers_n_jumps/01_train_remove_jumps_JFD/qsubs/
for runname in {1..10}
do
qsub ./q_JFD$runname.sh
done
