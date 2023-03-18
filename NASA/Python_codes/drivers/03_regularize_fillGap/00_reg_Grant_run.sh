#!/bin/bash

cd /home/hnoorazar/NASA/03_regularize_fillGap/qsubs/
for runname in {1..4}
do
qsub ./q_Grant$runname.sh
done
