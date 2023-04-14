#!/bin/bash

cd /home/h.noorazar/NASA/regionalStat/01_applyModels/qsubs

for runname in {1..192}
do
sbatch ./applyModels$runname.sh
done
