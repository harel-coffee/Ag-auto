#!/bin/bash

#PBS -l nodes=1:ppn=1,walltime=00:05:00
#PBS -l mem=1gb
#PBS -q fast
cd /home/hnoorazar/lagoon_codes/01_merge_runoff/sh
qsub 00_q_merge_run_bcc-csm1-1-m.sh
qsub 00_q_merge_run_GFDL-ESM2G.sh
qsub 00_q_merge_run_IPSL-CM5B-LR.sh
qsub 00_q_merge_run_bcc-csm1-1.sh
qsub 00_q_merge_run_GFDL-ESM2M.sh
qsub 00_q_merge_run_MIROC5.sh
qsub 00_q_merge_run_BNU-ESM.sh
qsub 00_q_merge_run_HadGEM2-CC365.sh
qsub 00_q_merge_run_MIROC-ESM-CHEM.sh
qsub 00_q_merge_run_CanESM2.sh
qsub 00_q_merge_run_HadGEM2-ES365.sh
qsub 00_q_merge_run_MRI-CGCM3.sh
qsub 00_q_merge_run_CCSM4.sh
qsub 00_q_merge_run_inmcm4.sh
qsub 00_q_merge_run_NorESM1-M.sh
qsub 00_q_merge_run_CNRM-CM5.sh
qsub 00_q_merge_run_IPSL-CM5A-LR.sh
qsub 00_q_merge_run_CSIRO.sh
qsub 00_q_merge_run_IPSL-CM5A-MR.sh

