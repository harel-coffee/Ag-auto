#!/bin/bash

cd /home/hnoorazar/analog_codes/03_find_analogs/models_separ/

for runname in q_bcc_45.sh q_bcc_85.sh q_BNU_45.sh q_BNU_85.sh q_CanESM2_45.sh q_CanESM2_85.sh q_CNRM_CM5_45.sh q_CNRM_CM5_85.sh q_GFDL_ESM2G_45.sh q_GFDL_ESM2G_85.sh  q_GFDL_ESM2M_45.sh q_GFDL_ESM2M_85.sh
do
qsub ./runname
done
