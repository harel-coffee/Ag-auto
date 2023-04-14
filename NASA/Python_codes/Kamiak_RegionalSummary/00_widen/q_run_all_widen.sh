#!/bin/bash

cd /home/h.noorazar/NASA/regionalStat/00_widen/qsubs

for runname in {1..4}
do
sbatch ./widenTemp$runname.sh
done
