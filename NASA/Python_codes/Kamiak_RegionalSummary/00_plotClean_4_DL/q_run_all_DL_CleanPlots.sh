#!/bin/bash

cd /home/h.noorazar/NASA/regionalStat/00_plotClean_4_DL/qsubs

for runname in {1..4}
do
sbatch ./DL_CleanPlots$runname.sh
done
