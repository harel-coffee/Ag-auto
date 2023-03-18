#!/bin/bash

cd /home/hnoorazar/NASA/07_clustering/qsubs/
for runname in {1..2}
do
qsub ./q_train_JFD_harmonize_$runname.sh
done
