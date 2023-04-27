#!/bin/bash

cd /home/hnoorazar/NASA/03_regularize_fillGap/qsubs/
for runname in {1..80}
do
qsub ./q_inters_JFD$runname.sh
done
