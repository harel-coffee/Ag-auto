#!/bin/bash

cd /home/hnoorazar/NASA/02_remove_outliers_n_jumps/01_intersect_remove_jumps_JFD/qsubs/
for runname in {1..81}
do
qsub ./q_JFD_post2008_$runname.sh
done
