#!/bin/bash

cd /home/hnoorazar/NASA/02_remove_outliers_n_jumps/00_Grant_remove_outliers/qsubs/
for runname in {1..4}
do
qsub ./q_$runname.sh
done
