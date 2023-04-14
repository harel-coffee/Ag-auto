#!/bin/bash
cd /home/h.noorazar/NASA/regionalStat/01_applyModels

#### size indeks smooth county model
outer=1
for size in large small
do 
  for indeks in EVI NDVI
  do
    for smooth in SG regular
    do
      for county in Adams Benton Franklin Grant Yakima Walla_Walla
      do
        for model in SVM RF DL KNN
        do
        cp applyModelsTemp.sh        ./qsubs/applyModels$outer.sh
        sed -i s/outer/"$outer"/g    ./qsubs/applyModels$outer.sh
        sed -i s/size/"$size"/g      ./qsubs/applyModels$outer.sh
        sed -i s/indeks/"$indeks"/g  ./qsubs/applyModels$outer.sh
        sed -i s/smooth/"$smooth"/g  ./qsubs/applyModels$outer.sh
        sed -i s/county/"$county"/g  ./qsubs/applyModels$outer.sh
        sed -i s/model/"$model"/g    ./qsubs/applyModels$outer.sh
        let "outer+=1"
        done
      done
    done
  done
done