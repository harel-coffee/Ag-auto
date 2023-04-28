#!/bin/bash

cd /home/hnoorazar/NASA/04_SG/qsubs/
for runname in {1..80}
do
qsub ./q_inters_JFD$runname.sh
done
