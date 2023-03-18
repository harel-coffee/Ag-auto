#!/bin/bash

cd /home/hnoorazar/NASA/05_snapshot_plots/qsubs/
for runname in {1..10}
do
qsub ./q_train_$runname.sh
done
