#!/share/apps/R-3.2.2_gcc/bin/Rscript
#library(chron)
library(data.table)
library(ggplot2)

#list.of.packages <- c("MESS")
#new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
#if(length(new.packages)) install.packages(new.packages)

#library(MESS)

#data_dir = getwd()
data_dir = "/data/hydro/users/giridhar/giridhar/codmoth_pop"
#filename <- paste0(data_dir, "/allData_grouped_counties.rds")
filename <- paste0(data_dir, "/allData_grouped_counties_rcp45.rds")
data <- data.table(readRDS(filename))

loc_grp = data.table(read.csv("LocationGroups2.csv"))
#loc_grp = loc_grp[1:15,]
loc_grp$latitude = as.numeric(loc_grp$latitude)
loc_grp$longitude = as.numeric(loc_grp$longitude)

data<-data[latitude %in% loc_grp$latitude & longitude %in% loc_grp$longitude]

theta = 0.2163108 + (2 * atan(0.9671396 * tan(0.00860 * (data$dayofyear - 186))))
phi = asin(0.39795 * cos(theta))
D = 24 - ((24/pi) * acos((sin(6 * pi / 180) + (sin(data$latitude * pi / 180) * sin(phi)))/(cos(data$latitude * pi / 180) * cos(phi))))
data$daylength = D

data$diapause = 102.6077 * exp(-exp(-(-1.306483) * (data$daylength - 16.95815)))
data$diapause1 = data$diapause
data[diapause1 > 100, diapause1 := 100]
data$enterDiap = (data$diapause1/100) * data$SumLarva
data$escapeDiap = data$SumLarva - data$enterDiap

sub = data
startingpopulationfortheyear<-1000
#generation1
sub[,LarvaGen1RelFraction := LarvaGen1/sum(LarvaGen1), by =list(year,ClimateScenario, latitude,longitude,ClimateGroup, CountyGroup) ]  ## Giridhar check this,,anything else to group by?
sub$AbsPopLarvaGen1<-sub$LarvaGen1RelFraction*startingpopulationfortheyear
sub$AbsPopLarvaGen1Diap<-sub$AbsPopLarvaGen1*sub$diapause1/100
sub$AbsPopLarvaGen1NonDiap<-sub$AbsPopLarvaGen1- sub$AbsPopLarvaGen1Diap

#generation2
sub[,LarvaGen2RelFraction := LarvaGen2/sum(LarvaGen2), by =list(year,ClimateScenario, latitude,longitude,ClimateGroup, CountyGroup) ]  ## Giridhar check this,,anything else to group by?
sub[,AbsPopLarvaGen2 := LarvaGen2RelFraction*sum(AbsPopLarvaGen1NonDiap)*3.9, by =list(year,ClimateScenario, latitude,longitude,ClimateGroup, CountyGroup) ]  ## Giridhar check this,,anything else to group by?
sub$AbsPopLarvaGen2Diap<-sub$AbsPopLarvaGen2*sub$diapause1/100
sub$AbsPopLarvaGen2NonDiap<-sub$AbsPopLarvaGen2- sub$AbsPopLarvaGen2Diap

#generation3
sub[,LarvaGen3RelFraction := LarvaGen3/sum(LarvaGen3), by =list(year,ClimateScenario, latitude,longitude,ClimateGroup, CountyGroup) ]  ## Giridhar check this,,anything else to group by?
sub[,AbsPopLarvaGen3 := LarvaGen3RelFraction*sum(AbsPopLarvaGen2NonDiap)*3.9, by =list(year,ClimateScenario, latitude,longitude,ClimateGroup, CountyGroup) ]  ## Giridhar check this,,anything else to group by?
sub$AbsPopLarvaGen3Diap<-sub$AbsPopLarvaGen3*sub$diapause1/100
sub$AbsPopLarvaGen3NonDiap<-sub$AbsPopLarvaGen3- sub$AbsPopLarvaGen3Diap

#generation4
sub[,LarvaGen4RelFraction := LarvaGen4/sum(LarvaGen4), by =list(year,ClimateScenario, latitude,longitude,ClimateGroup, CountyGroup) ]  ## Giridhar check this,,anything else to group by?
sub[,AbsPopLarvaGen4 := LarvaGen4RelFraction*sum(AbsPopLarvaGen3NonDiap)*3.9, by =list(year,ClimateScenario, latitude,longitude,ClimateGroup, CountyGroup) ]  ## Giridhar check this,,anything else to group by?
sub$AbsPopLarvaGen4Diap<-sub$AbsPopLarvaGen4*sub$diapause1/100
sub$AbsPopLarvaGen4NonDiap<-sub$AbsPopLarvaGen4- sub$AbsPopLarvaGen4Diap

### get totals similar to Sum Larva column , but abs numbers
sub$AbsPopTotal<-sub$AbsPopLarvaGen1+sub$AbsPopLarvaGen2+sub$AbsPopLarvaGen3+sub$AbsPopLarvaGen4
sub$AbsPopDiap<-sub$AbsPopLarvaGen1Diap+sub$AbsPopLarvaGen2Diap+sub$AbsPopLarvaGen3Diap+sub$AbsPopLarvaGen4Diap
sub$AbsPopNonDiap<-sub$AbsPopLarvaGen1NonDiap+sub$AbsPopLarvaGen2NonDiap+sub$AbsPopLarvaGen3NonDiap+sub$AbsPopLarvaGen4NonDiap

sub1 = subset(sub, select = c("latitude", "longitude", "County", "CountyGroup", "ClimateScenario", "ClimateGroup", "year", "dayofyear", "CumDDinF", "SumLarva", "enterDiap", "escapeDiap", "AbsPopTotal","AbsPopNonDiap","AbsPopDiap"))
#sub1 = sub1[, .(RelLarvaPop = mean(SumLarva), RelDiap = mean(enterDiap), RelNonDiap = mean(escapeDiap), AbsLarvaPop = mean(AbsPopTotal), AbsDiap = mean(AbsPopDiap), AbsNonDiap = mean(AbsPopNonDiap), CumulativeDDF = mean(CumDDinF)), by = c("ClimateGroup", "CountyGroup", "latitude", "longitude", "dayofyear")]  ## Check with Kirti regarding CumDDinF

sub1 = sub1[, .(RelLarvaPop = mean(SumLarva), RelDiap = mean(enterDiap), RelNonDiap = mean(escapeDiap), AbsLarvaPop = mean(AbsPopTotal), AbsDiap = mean(AbsPopDiap), AbsNonDiap = mean(AbsPopNonDiap), CumulativeDDF = mean(CumDDinF)), by = c("ClimateGroup", "CountyGroup", "dayofyear")]  ## Check with Kirti regarding CumDDinF

#saveRDS(sub1, paste0(data_dir, "/", "diapause_plot_data.rds"))

RelData = subset(sub1, select = c("ClimateGroup", "CountyGroup", "dayofyear", "CumulativeDDF", "RelLarvaPop", "RelDiap", "RelNonDiap"))
RelData = melt(RelData, id = c("ClimateGroup", "CountyGroup", "dayofyear", "CumulativeDDF"))
#RelData[, Group := as.character()]
RelData$Group = "0"
temp1 = RelData[variable %in% c("RelLarvaPop", "RelDiap")]
temp2 = RelData[variable %in% c("RelLarvaPop", "RelNonDiap")]
temp1$Group = "Relative Larva Pop Vs Diapaused"
temp2$Group = "Relative Larva Pop Vs NonDiapaused"
RelData = rbind(temp1, temp2)

#saveRDS(RelData, paste0(data_dir, "/", "diapause_rel_data.rds"))
saveRDS(RelData, paste0(data_dir, "/", "diapause_rel_data_rcp45.rds"))

AbsData = subset(sub1, select = c("ClimateGroup", "CountyGroup", "dayofyear", "CumulativeDDF", "AbsLarvaPop", "AbsDiap", "AbsNonDiap"))
AbsData = melt(AbsData, id = c("ClimateGroup", "CountyGroup", "dayofyear", "CumulativeDDF"))
#AbsData[, Group := as.character()]
AbsData$Group = "0"
temp1 = AbsData[variable %in% c("AbsLarvaPop", "AbsDiap")]
temp2 = AbsData[variable %in% c("AbsLarvaPop", "AbsNonDiap")]
temp1$Group = "Absolute Larva Pop Vs Diapaused"
temp2$Group = "Absolute Larva Pop Vs NonDiapaused"
AbsData = rbind(temp1, temp2)

#saveRDS(AbsData, paste0(data_dir, "/", "diapause_abs_data.rds"))
saveRDS(AbsData, paste0(data_dir, "/", "diapause_abs_data_rcp45.rds"))

#saveRDS(sub1, paste0(data_dir, "/", "diapause_map_data_rcp45.rds"))

#sub2 = sub1[, .(RelPctDiap = (auc(RelDiap)/auc(RelLarvaPop))*100, RelPctNonDiap = (auc(RelNonDiap)/auc(RelLarvaPop))*100, AbsPctDiap = (auc(AbsDiap)/auc(AbsLarvaPop))*100, AbsPctNonDiap = (auc(AbsNonDiap)/auc(AbsLarvaPop))*100), by = c("ClimateGroup", "CountyGroup", "latitude", "longitude")]
#
#CodMothParams <- read.table(paste0(data_dir, "CodlingMothparameters.txt"),header=TRUE,sep=",")
#
#sub2$RelPctDiapGen1 = sub1[CumulativeDDF >= CodMothParams[5,5] & CumulativeDDF < CodMothParams[5,6], .(pct = (auc(RelDiap)/auc(RelLarvaDiap))*100), by = c("ClimateGroup", "CountyGroup", "latitude", "longitude")]$pct
#sub2$RelPctDiapGen2 = sub1[CumulativeDDF >= CodMothParams[6,5] & CumulativeDDF < CodMothParams[6,6], .(pct = (auc(RelDiap)/auc(RelLarvaDiap))*100), by = c("ClimateGroup", "CountyGroup", "latitude", "longitude")]$pct
#sub2$RelPctDiapGen3 = sub1[CumulativeDDF >= CodMothParams[7,5] & CumulativeDDF < CodMothParams[7,6], .(pct = (auc(RelDiap)/auc(RelLarvaDiap))*100), by = c("ClimateGroup", "CountyGroup", "latitude", "longitude")]$pct
#sub2$RelPctDiapGen4 = sub1[CumulativeDDF >= CodMothParams[8,5] & CumulativeDDF < CodMothParams[8,6], .(pct = (auc(RelDiap)/auc(RelLarvaDiap))*100), by = c("ClimateGroup", "CountyGroup", "latitude", "longitude")]$pct
#
#sub2$RelPctNonDiapGen1 = sub1[CumulativeDDF >= CodMothParams[5,5] & CumulativeDDF < CodMothParams[5,6], .(pct = (auc(RelNonDiap)/auc(RelLarvaDiap))*100), by = c("ClimateGroup", "CountyGroup", "latitude", "longitude")]$pct
#sub2$RelPctNonDiapGen2 = sub1[CumulativeDDF >= CodMothParams[6,5] & CumulativeDDF < CodMothParams[6,6], .(pct = (auc(RelNonDiap)/auc(RelLarvaDiap))*100), by = c("ClimateGroup", "CountyGroup", "latitude", "longitude")]$pct
#sub2$RelPctNonDiapGen3 = sub1[CumulativeDDF >= CodMothParams[7,5] & CumulativeDDF < CodMothParams[7,6], .(pct = (auc(RelNonDiap)/auc(RelLarvaDiap))*100), by = c("ClimateGroup", "CountyGroup", "latitude", "longitude")]$pct
#sub2$RelPctNonDiapGen4 = sub1[CumulativeDDF >= CodMothParams[8,5] & CumulativeDDF < CodMothParams[8,6], .(pct = (auc(RelNonDiap)/auc(RelLarvaDiap))*100), by = c("ClimateGroup", "CountyGroup", "latitude", "longitude")]$pct
#
#sub2$AbsPctDiapGen1 = sub1[CumulativeDDF >= CodMothParams[5,5] & CumulativeDDF < CodMothParams[5,6], .(pct = (auc(AbsDiap)/auc(AbsLarvaDiap))*100), by = c("ClimateGroup", "CountyGroup", "latitude", "longitude")]$pct
#sub2$AbsPctDiapGen2 = sub1[CumulativeDDF >= CodMothParams[6,5] & CumulativeDDF < CodMothParams[6,6], .(pct = (auc(AbsDiap)/auc(AbsLarvaDiap))*100), by = c("ClimateGroup", "CountyGroup", "latitude", "longitude")]$pct
#sub2$AbsPctDiapGen3 = sub1[CumulativeDDF >= CodMothParams[7,5] & CumulativeDDF < CodMothParams[7,6], .(pct = (auc(AbsDiap)/auc(AbsLarvaDiap))*100), by = c("ClimateGroup", "CountyGroup", "latitude", "longitude")]$pct
#sub2$AbsPctDiapGen4 = sub1[CumulativeDDF >= CodMothParams[8,5] & CumulativeDDF < CodMothParams[8,6], .(pct = (auc(AbsDiap)/auc(AbsLarvaDiap))*100), by = c("ClimateGroup", "CountyGroup", "latitude", "longitude")]$pct
#
#sub2$AbsPctNonDiapGen1 = sub1[CumulativeDDF >= CodMothParams[5,5] & CumulativeDDF < CodMothParams[5,6], .(pct = (auc(AbsNonDiap)/auc(AbsLarvaDiap))*100), by = c("ClimateGroup", "CountyGroup", "latitude", "longitude")]$pct
#sub2$AbsPctNonDiapGen2 = sub1[CumulativeDDF >= CodMothParams[6,5] & CumulativeDDF < CodMothParams[6,6], .(pct = (auc(AbsNonDiap)/auc(AbsLarvaDiap))*100), by = c("ClimateGroup", "CountyGroup", "latitude", "longitude")]$pct
#sub2$AbsPctNonDiapGen3 = sub1[CumulativeDDF >= CodMothParams[7,5] & CumulativeDDF < CodMothParams[7,6], .(pct = (auc(AbsNonDiap)/auc(AbsLarvaDiap))*100), by = c("ClimateGroup", "CountyGroup", "latitude", "longitude")]$pct
#sub2$AbsPctNonDiapGen4 = sub1[CumulativeDDF >= CodMothParams[8,5] & CumulativeDDF < CodMothParams[8,6], .(pct = (auc(AbsNonDiap)/auc(AbsLarvaDiap))*100), by = c("ClimateGroup", "CountyGroup", "latitude", "longitude")]$pct
#

#sub2 = sub1[, .(RelPctDiap = (auc(CumulativeDDF,RelDiap)/auc(CumulativeDDF,RelLarvaPop))*100, RelPctNonDiap = (auc(CumulativeDDF,RelNonDiap)/auc(CumulativeDDF,RelLarvaPop))*100, AbsPctDiap = (auc(CumulativeDDF,AbsDiap)/auc(CumulativeDDF,AbsLarvaPop))*100, AbsPctNonDiap = (auc(CumulativeDDF,AbsNonDiap)/auc(CumulativeDDF,AbsLarvaPop))*100), by = c("ClimateGroup", "CountyGroup", "latitude", "longitude")]
#
#CodMothParams <- read.table(paste0(data_dir, "CodlingMothparameters.txt"),header=TRUE,sep=",")
#
#sub2 = merge(sub2, sub1[CumulativeDDF >= CodMothParams[5,5] & CumulativeDDF < CodMothParams[5,6], .(RelPctDiapGen1 = (auc(CumulativeDDF,RelDiap)/auc(CumulativeDDF,RelLarvaPop))*100), by = c("ClimateGroup", "CountyGroup", "latitude", "longitude")], by = c("ClimateGroup", "CountyGroup", "latitude", "longitude"), all.x = TRUE)
#sub2 = merge(sub2, sub1[CumulativeDDF >= CodMothParams[6,5] & CumulativeDDF < CodMothParams[6,6], .(RelPctDiapGen2 = (auc(CumulativeDDF,RelDiap)/auc(CumulativeDDF,RelLarvaPop))*100), by = c("ClimateGroup", "CountyGroup", "latitude", "longitude")], by = c("ClimateGroup", "CountyGroup", "latitude", "longitude"), all.x = TRUE)
#sub2 = merge(sub2, sub1[CumulativeDDF >= CodMothParams[7,5] & CumulativeDDF < CodMothParams[7,6], .(RelPctDiapGen3 = (auc(CumulativeDDF,RelDiap)/auc(CumulativeDDF,RelLarvaPop))*100), by = c("ClimateGroup", "CountyGroup", "latitude", "longitude")], by = c("ClimateGroup", "CountyGroup", "latitude", "longitude"), all.x = TRUE)
#sub2 = merge(sub2, sub1[CumulativeDDF >= CodMothParams[8,5] & CumulativeDDF < CodMothParams[8,6], .(RelPctDiapGen4 = (auc(CumulativeDDF,RelDiap)/auc(CumulativeDDF,RelLarvaPop))*100), by = c("ClimateGroup", "CountyGroup", "latitude", "longitude")], by = c("ClimateGroup", "CountyGroup", "latitude", "longitude"), all.x = TRUE)
##sub2$RelPctDiapGen1 = sub1[CumulativeDDF >= CodMothParams[5,5] & CumulativeDDF < CodMothParams[5,6], .(pct = if(auc(CumulativeDDF,RelLarvaPop) == 0) NA else (auc(CumulativeDDF,RelDiap)/auc(CumulativeDDF,RelLarvaPop))*100), by = c("ClimateGroup", "CountyGroup", "latitude", "longitude")]$pct
##sub2$RelPctDiapGen2 = sub1[CumulativeDDF >= CodMothParams[6,5] & CumulativeDDF < CodMothParams[6,6], .(pct = if(auc(CumulativeDDF,RelLarvaPop) == 0) NA else (auc(CumulativeDDF,RelDiap)/auc(CumulativeDDF,RelLarvaPop))*100), by = c("ClimateGroup", "CountyGroup", "latitude", "longitude")]$pct
##sub2$RelPctDiapGen3 = sub1[CumulativeDDF >= CodMothParams[7,5] & CumulativeDDF < CodMothParams[7,6], .(pct = if(auc(CumulativeDDF,RelLarvaPop) == 0) NA else (auc(CumulativeDDF,RelDiap)/auc(CumulativeDDF,RelLarvaPop))*100), by = c("ClimateGroup", "CountyGroup", "latitude", "longitude")]$pct
##sub2$RelPctDiapGen4 = sub1[CumulativeDDF >= CodMothParams[8,5] & CumulativeDDF < CodMothParams[8,6], .(pct = if(auc(CumulativeDDF,RelLarvaPop) == 0) NA else (auc(CumulativeDDF,RelDiap)/auc(CumulativeDDF,RelLarvaPop))*100), by = c("ClimateGroup", "CountyGroup", "latitude", "longitude")]$pct
#
#sub2 = merge(sub2, sub1[CumulativeDDF >= CodMothParams[5,5] & CumulativeDDF < CodMothParams[5,6], .(RelPctNonDiapGen1 = (auc(CumulativeDDF,RelNonDiap)/auc(CumulativeDDF,RelLarvaPop))*100), by = c("ClimateGroup", "CountyGroup", "latitude", "longitude")], by = c("ClimateGroup", "CountyGroup", "latitude", "longitude"), all.x = TRUE)
#sub2 = merge(sub2, sub1[CumulativeDDF >= CodMothParams[6,5] & CumulativeDDF < CodMothParams[6,6], .(RelPctNonDiapGen2 = (auc(CumulativeDDF,RelNonDiap)/auc(CumulativeDDF,RelLarvaPop))*100), by = c("ClimateGroup", "CountyGroup", "latitude", "longitude")], by = c("ClimateGroup", "CountyGroup", "latitude", "longitude"), all.x = TRUE)
#sub2 = merge(sub2, sub1[CumulativeDDF >= CodMothParams[7,5] & CumulativeDDF < CodMothParams[7,6], .(RelPctNonDiapGen3 = (auc(CumulativeDDF,RelNonDiap)/auc(CumulativeDDF,RelLarvaPop))*100), by = c("ClimateGroup", "CountyGroup", "latitude", "longitude")], by = c("ClimateGroup", "CountyGroup", "latitude", "longitude"), all.x = TRUE)
#sub2 = merge(sub2, sub1[CumulativeDDF >= CodMothParams[8,5] & CumulativeDDF < CodMothParams[8,6], .(RelPctNonDiapGen4 = (auc(CumulativeDDF,RelNonDiap)/auc(CumulativeDDF,RelLarvaPop))*100), by = c("ClimateGroup", "CountyGroup", "latitude", "longitude")], by = c("ClimateGroup", "CountyGroup", "latitude", "longitude"), all.x = TRUE)
##sub2$RelPctNonDiapGen1 = sub1[CumulativeDDF >= CodMothParams[5,5] & CumulativeDDF < CodMothParams[5,6], .(pct = (auc(CumulativeDDF,RelNonDiap)/auc(CumulativeDDF,RelLarvaPop))*100), by = c("ClimateGroup", "CountyGroup", "latitude", "longitude")]$pct
##sub2$RelPctNonDiapGen2 = sub1[CumulativeDDF >= CodMothParams[6,5] & CumulativeDDF < CodMothParams[6,6], .(pct = (auc(CumulativeDDF,RelNonDiap)/auc(CumulativeDDF,RelLarvaPop))*100), by = c("ClimateGroup", "CountyGroup", "latitude", "longitude")]$pct
##sub2$RelPctNonDiapGen3 = sub1[CumulativeDDF >= CodMothParams[7,5] & CumulativeDDF < CodMothParams[7,6], .(pct = (auc(CumulativeDDF,RelNonDiap)/auc(CumulativeDDF,RelLarvaPop))*100), by = c("ClimateGroup", "CountyGroup", "latitude", "longitude")]$pct
##sub2$RelPctNonDiapGen4 = sub1[CumulativeDDF >= CodMothParams[8,5] & CumulativeDDF < CodMothParams[8,6], .(pct = (auc(CumulativeDDF,RelNonDiap)/auc(CumulativeDDF,RelLarvaPop))*100), by = c("ClimateGroup", "CountyGroup", "latitude", "longitude")]$pct
#
# sub2 = merge(sub2, sub1[CumulativeDDF >= CodMothParams[5,5] & CumulativeDDF < CodMothParams[5,6], .(AbsPctDiapGen1 = (auc(CumulativeDDF,AbsDiap)/auc(CumulativeDDF,AbsLarvaPop))*100), by = c("ClimateGroup", "CountyGroup", "latitude", "longitude")], by = c("ClimateGroup", "CountyGroup", "latitude", "longitude"), all.x = TRUE)
# sub2 = merge(sub2, sub1[CumulativeDDF >= CodMothParams[6,5] & CumulativeDDF < CodMothParams[6,6], .(AbsPctDiapGen2 = (auc(CumulativeDDF,AbsDiap)/auc(CumulativeDDF,AbsLarvaPop))*100), by = c("ClimateGroup", "CountyGroup", "latitude", "longitude")], by = c("ClimateGroup", "CountyGroup", "latitude", "longitude"), all.x = TRUE)
# sub2 = merge(sub2, sub1[CumulativeDDF >= CodMothParams[7,5] & CumulativeDDF < CodMothParams[7,6], .(AbsPctDiapGen3 = (auc(CumulativeDDF,AbsDiap)/auc(CumulativeDDF,AbsLarvaPop))*100), by = c("ClimateGroup", "CountyGroup", "latitude", "longitude")], by = c("ClimateGroup", "CountyGroup", "latitude", "longitude"), all.x = TRUE)
# sub2 = merge(sub2, sub1[CumulativeDDF >= CodMothParams[8,5] & CumulativeDDF < CodMothParams[8,6], .(AbsPctDiapGen4 = (auc(CumulativeDDF,AbsDiap)/auc(CumulativeDDF,AbsLarvaPop))*100), by = c("ClimateGroup", "CountyGroup", "latitude", "longitude")], by = c("ClimateGroup", "CountyGroup", "latitude", "longitude"), all.x = TRUE)
##sub2$AbsPctDiapGen1 = sub1[CumulativeDDF >= CodMothParams[5,5] & CumulativeDDF < CodMothParams[5,6], .(pct = (auc(CumulativeDDF,AbsDiap)/auc(CumulativeDDF,AbsLarvaPop))*100), by = c("ClimateGroup", "CountyGroup", "latitude", "longitude")]$pct
##sub2$AbsPctDiapGen2 = sub1[CumulativeDDF >= CodMothParams[6,5] & CumulativeDDF < CodMothParams[6,6], .(pct = (auc(CumulativeDDF,AbsDiap)/auc(CumulativeDDF,AbsLarvaPop))*100), by = c("ClimateGroup", "CountyGroup", "latitude", "longitude")]$pct
##sub2$AbsPctDiapGen3 = sub1[CumulativeDDF >= CodMothParams[7,5] & CumulativeDDF < CodMothParams[7,6], .(pct = (auc(CumulativeDDF,AbsDiap)/auc(CumulativeDDF,AbsLarvaPop))*100), by = c("ClimateGroup", "CountyGroup", "latitude", "longitude")]$pct
##sub2$AbsPctDiapGen4 = sub1[CumulativeDDF >= CodMothParams[8,5] & CumulativeDDF < CodMothParams[8,6], .(pct = (auc(CumulativeDDF,AbsDiap)/auc(CumulativeDDF,AbsLarvaPop))*100), by = c("ClimateGroup", "CountyGroup", "latitude", "longitude")]$pct
#
#sub2 = merge(sub2, sub1[CumulativeDDF >= CodMothParams[5,5] & CumulativeDDF < CodMothParams[5,6], .(AbsPctNonDiapGen1 = (auc(CumulativeDDF,AbsNonDiap)/auc(CumulativeDDF,AbsLarvaPop))*100), by = c("ClimateGroup", "CountyGroup", "latitude", "longitude")], by = c("ClimateGroup", "CountyGroup", "latitude", "longitude"), all.x = TRUE)
#sub2 = merge(sub2, sub1[CumulativeDDF >= CodMothParams[6,5] & CumulativeDDF < CodMothParams[6,6], .(AbsPctNonDiapGen2 = (auc(CumulativeDDF,AbsNonDiap)/auc(CumulativeDDF,AbsLarvaPop))*100), by = c("ClimateGroup", "CountyGroup", "latitude", "longitude")], by = c("ClimateGroup", "CountyGroup", "latitude", "longitude"), all.x = TRUE)
#sub2 = merge(sub2, sub1[CumulativeDDF >= CodMothParams[7,5] & CumulativeDDF < CodMothParams[7,6], .(AbsPctNonDiapGen3 = (auc(CumulativeDDF,AbsNonDiap)/auc(CumulativeDDF,AbsLarvaPop))*100), by = c("ClimateGroup", "CountyGroup", "latitude", "longitude")], by = c("ClimateGroup", "CountyGroup", "latitude", "longitude"), all.x = TRUE)
#sub2 = merge(sub2, sub1[CumulativeDDF >= CodMothParams[8,5] & CumulativeDDF < CodMothParams[8,6], .(AbsPctNonDiapGen4 = (auc(CumulativeDDF,AbsNonDiap)/auc(CumulativeDDF,AbsLarvaPop))*100), by = c("ClimateGroup", "CountyGroup", "latitude", "longitude")], by = c("ClimateGroup", "CountyGroup", "latitude", "longitude"), all.x = TRUE)
##sub2$AbsPctNonDiapGen1 = sub1[CumulativeDDF >= CodMothParams[5,5] & CumulativeDDF < CodMothParams[5,6], .(pct = (auc(CumulativeDDF,AbsNonDiap)/auc(CumulativeDDF,AbsLarvaPop))*100), by = c("ClimateGroup", "CountyGroup", "latitude", "longitude")]$pct
##sub2$AbsPctNonDiapGen2 = sub1[CumulativeDDF >= CodMothParams[6,5] & CumulativeDDF < CodMothParams[6,6], .(pct = (auc(CumulativeDDF,AbsNonDiap)/auc(CumulativeDDF,AbsLarvaPop))*100), by = c("ClimateGroup", "CountyGroup", "latitude", "longitude")]$pct
##sub2$AbsPctNonDiapGen3 = sub1[CumulativeDDF >= CodMothParams[7,5] & CumulativeDDF < CodMothParams[7,6], .(pct = (auc(CumulativeDDF,AbsNonDiap)/auc(CumulativeDDF,AbsLarvaPop))*100), by = c("ClimateGroup", "CountyGroup", "latitude", "longitude")]$pct
##sub2$AbsPctNonDiapGen4 = sub1[CumulativeDDF >= CodMothParams[8,5] & CumulativeDDF < CodMothParams[8,6], .(pct = (auc(CumulativeDDF,AbsNonDiap)/auc(CumulativeDDF,AbsLarvaPop))*100), by = c("ClimateGroup", "CountyGroup", "latitude", "longitude")]$pct
#
#saveRDS(sub2, paste0(data_dir, "/", "diapause_map_data_rcp45.rds"))
#
##
###saveRDS(data, paste0(data_dir, "/", "allData_grouped_counties.rds"))
