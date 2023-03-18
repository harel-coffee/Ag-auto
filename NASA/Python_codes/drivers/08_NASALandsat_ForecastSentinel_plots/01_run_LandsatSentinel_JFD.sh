#!/bin/bash

cd /home/hnoorazar/NASA/08_NASALandsat_ForecastSentinel_plots/qsubs/
for runname in {1..4}
do
qsub ./q_exactEval_JFD$runname.sh
done
