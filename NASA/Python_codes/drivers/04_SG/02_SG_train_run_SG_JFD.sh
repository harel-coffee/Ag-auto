#!/bin/bash

cd /home/hnoorazar/NASA/04_SG/qsubs/
for runname in {1..8}
do
qsub ./q_train_JFD$runname.sh
done
