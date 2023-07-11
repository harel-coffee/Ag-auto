#!/bin/bash

cd /home/hnoorazar/NASA/02_remove_outliers_n_jumps/00_intersect_remove_outliers/qsubs/

for runname in {1..100}
do
  qsub ./q_pre2008_mergeFirst$runname.sh
done
