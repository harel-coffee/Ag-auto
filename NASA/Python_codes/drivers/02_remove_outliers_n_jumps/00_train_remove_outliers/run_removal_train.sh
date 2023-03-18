#!/bin/bash

cd /home/hnoorazar/NASA/02_remove_outliers_n_jumps/00_train_remove_outliers/qsubs/
for runname in {1..10}
do
qsub ./q_$runname.sh
done
