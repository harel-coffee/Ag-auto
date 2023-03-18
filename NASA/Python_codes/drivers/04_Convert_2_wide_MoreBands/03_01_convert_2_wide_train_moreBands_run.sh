#!/bin/bash

cd /home/hnoorazar/NASA/03_01_widen_MoreBands/qsubs/
for runname in {1..6}
do
qsub ./q_train_moreBands_widen$runname.sh
done
