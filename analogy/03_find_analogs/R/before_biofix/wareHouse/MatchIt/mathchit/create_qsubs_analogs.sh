#!/bin/bash

for model_carbon in bcc_csm1_1_m/rcp45 bcc_csm1_1_m/rcp85 BNU_ESM/rcp45 BNU_ESM/rcp85 CanESM2/rcp45 CanESM2/rcp85 CNRM_CM5/rcp45 CNRM_CM5/rcp85 GFDL_ESM2G/rcp45 GFDL_ESM2G/rcp85 GFDL_ESM2M/rcp45 GFDL_ESM2M/rcp85
do
rm -rf ./$model_carbon
mkdir ./$model_carbon
cp ./qsub_temp_analogs ./$model_carbon/qsub_set1
sed -i s/thisjobname/"$model_carbon"/g ./$model_carbon/qsub_set1
cd ./$model_carbon
for ((i = 1; i <= 295; i++)) 
do
cp qsub_set1 qsub_set$i
sed -i s/set1/"set$i"/g ./qsub_set$i
done
cd ..
done


dir_list=()
while IFS= read -d $'\0' -r file ; do
dir_list=("${dir_list[@]}" "$file")
done < <(find /data/hydro/users/Hossein/analog/local/ready_features/broken_down -mindepth 2 -maxdepth 2 -type f -print0)

echo
echo "${dir_list[@]}"
echo