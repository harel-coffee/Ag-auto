#!/bin/bash
cd /home/hnoorazar/lagoon_codes/01_run_offs/01_cum_runs/

outer=1
for file in run_offsbcc_csm1_1_m.rds run_offsGFDL_ESM2G.rds run_offsIPSL_CM5B_LR.rds run_offsbcc_csm1_1.rds run_offsCSIRO_Mk3_6_0.rds run_offsIPSL_CM5A_MR.rds run_offsCNRM_CM5.rds run_offsIPSL_CM5A_LR.rds run_offs_observed.rds run_offsCCSM4.rds run_offsinmcm4.rds run_offsNorESM1_M.rds  run_offsCanESM2.rds run_offsHadGEM2_ES365.rds run_offsMRI_CGCM3.rds run_offsBNU_ESM.rds run_offsHadGEM2_CC365.rds  run_offsMIROC_ESM_CHEM.rds run_offsGFDL_ESM2M.rds run_offsMIROC5.rds
do
  cp template_ann.sh        ./qsubs/q_ann_$outer.sh
  sed -i s/outer/"$outer"/g ./qsubs/q_ann_$outer.sh
  sed -i s/fileN/"$file"/g  ./qsubs/q_ann_$outer.sh

  cp template_chunk.sh      ./qsubs/q_chunk_$outer.sh
  sed -i s/outer/"$outer"/g ./qsubs/q_chunk_$outer.sh
  sed -i s/fileN/"$file"/g  ./qsubs/q_chunk_$outer.sh
  
  cp template_wtr_yr.sh     ./qsubs/q_wtr_yr_$outer.sh
  sed -i s/outer/"$outer"/g ./qsubs/q_wtr_yr_$outer.sh
  sed -i s/fileN/"$file"/g  ./qsubs/q_wtr_yr_$outer.sh

  cp template_month.sh      ./qsubs/q_month_$outer.sh
  sed -i s/outer/"$outer"/g ./qsubs/q_month_$outer.sh
  sed -i s/fileN/"$file"/g  ./qsubs/q_month_$outer.sh

  let "outer+=1" 
done  
