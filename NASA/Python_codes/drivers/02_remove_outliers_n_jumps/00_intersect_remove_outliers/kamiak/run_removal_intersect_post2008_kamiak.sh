#!/bin/bash

cd /home/h.noorazar/NASA/02_remove_outliers_n_jumps/00_intersect_remove_outliers/qsubs/

for runname in {1..10}
do
  qsub ./q_post2008_$runname.sh
done
