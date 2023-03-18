#!/bin/bash

cd /home/hnoorazar/NASA/03_regularize_fillGap/qsubs/
for runname in {1..24}
do
qsub ./q_train_moreBands$runname.sh
done
