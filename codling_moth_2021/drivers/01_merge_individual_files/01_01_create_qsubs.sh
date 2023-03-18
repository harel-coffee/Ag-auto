#!/bin/bash
cd /home/hnoorazar/codling_moth_2021/drivers/01_merge_individual_files/

outer=1
for sub_dir in 00_LO_CMPOP 00_LM_CMPOP 00_LO_CM 00_LM_CM
do
  for emission in observed rcp45 rcp85
  do
    cp 01_00_merge_template.sh      ./qsubs/q_$outer.sh
    sed -i s/outer/"$outer"/g       ./qsubs/q_$outer.sh
    sed -i s/sub_dir/"$sub_dir"/g   ./qsubs/q_$outer.sh
    sed -i s/emission/"$emission"/g ./qsubs/q_$outer.sh
    let "outer+=1"
  done
done



