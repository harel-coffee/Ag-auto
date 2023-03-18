#!/bin/bash

cd /home/hnoorazar/NASA/06_2crop_acr/qsubs/
for runname in {1..2}
do
qsub ./q_train_JFD_$runname.sh
done
