rm(list=ls())
library(shiny)
library(shinydashboard)
library(htmlwidgets)
library(webshot)
library(shinyBS)
library(rgdal)    # for readOGR and others
library(maps)
library(sp)       # for spatial objects
library(leaflet)  # for interactive maps (NOT leafletR here)
library(dplyr)    # for working with data frames
library(ggplot2)  # for plotting
library(data.table)
library(reshape2)
library(RColorBrewer)

source_path = "/Users/hn/Documents/GitHub/Kirti/codling_moth/code/laptop_codes/map_functions.R"
source(source_path)

# data_dir <- "/Users/hn/Desktop/Kirti/check_point/CAHNR/"
data_dir <- "/Users/hn/Desktop/Kirti/check_point/CAHNR/2015/"
setwd(data_dir)

d = data.table(readRDS(paste0(data_dir,"combinedData.rds")))
d$timeFrame <-as.factor(d$timeFrame)
d$timeFrame <- factor(d$timeFrame, levels = levels(d$timeFrame)[c(4,1,2,3)])

d_rcp45 = data.table(readRDS(paste0(data_dir,"combinedData_rcp45.rds")))

### The followings were written for Giridhar's messed up files!
# names(d_rcp45)[names(d_rcp45) == "ClimateGroup"] = "timeFrame"
# d_rcp45$location = paste0(d_rcp45$latitude, "_", d_rcp45$longitude)
# cols <- colnames(d)
# d_rcp45 <- subset(d_rcp45, select=cols)

RdBu_reverse <- rev(brewer.pal(11, "RdBu"))
################################################
################
################ Egg Hatch Part
################
################################################

################
############################ Pest Risk
################
type_risk = "L"
cg_risks = c("Historical", "2040's", "2060's", "2080's")
gen_risks = c("Gen3", "Gen4")
future_versions = c("rcp45", "rcp85")
percent_risks = c(.25, .5, .75)

for (climate_group in cg_risks){
  for (future_version in future_versions){
    if (climate_group == "Historical"){future_version = "rcp85"}
    if(future_version == "rcp45") { data = d_rcp45} else { data = d }
    
    for (gen_risk in gen_risks){
      for (percent_risk in percent_risks){
        genPct = paste0(type_risk, gen_risk, "_", percent_risk)
        layerlist = levels(data$timeFrame)
        freq_data = subset(data, !is.na(timeFrame) & timeFrame == climate_group, 
                           select = c(timeFrame, year, location, get(genPct)))
        freq_data_melted = melt(freq_data, id.vars = c("timeFrame", "location", "year"), na.rm = FALSE)
        f1 = freq_data_melted[, .(years_range = (max(year) - min(year) + 1)), 
                               by = list(timeFrame, location)][order(timeFrame, location)]
        f2 = freq_data_melted[complete.cases(freq_data_melted$value), 
                      .(years_freq = uniqueN(year)), 
                      by = list(timeFrame, location)][order(timeFrame, location)]
        # left join - merge both tables
        f = merge(f1, f2, by = c("timeFrame", "location"), all.x = TRUE)
        # replace na values by 0
        f[is.na(years_freq), years_freq := 0]
        f$percentage = (f$years_freq / f$years_range) * 100
        riskGen = list(hist   = subset(f, timeFrame == layerlist[1]),
                       `2040` = subset(f, timeFrame == layerlist[2]),
                       `2060` = subset(f, timeFrame == layerlist[3]),
                       `2080` = subset(f, timeFrame == layerlist[4]))
        GenMap <- constructMap(riskGen, layerlist, palColumn = "percentage", 
                               legendVals = seq(0, 100), 
                               HTML("Percentage of<br />Years Occurred"),
                               RdBu_reverse)
        GenMap
        saveWidget(GenMap, "temp.html", selfcontained = FALSE)
        file_name = paste0("pest_risk_", climate_group, "_", 
                           future_version, "_", gen_risk, "_", 
                           percent_risk,".pdf")
        webshot("temp.html", file = file_name, cliprect = "viewport")
      }
    }
  }
}
################
############################ Median Day of Year
################
domainVal = seq(110, 345)
domainVal = seq(10, 400) # MDoY_1 folder
domainVal = seq(90, 365) # MDoY_2 folder
type = "L"

cg_larvae_med_doy = c("Historical", "2040's", "2060's", "2080's")
generations = c("Gen1", "Gen2", "Gen3", "Gen4")
future_versions = c("rcp45", "rcp85")
percentages = c("0.25", "0.5", "0.75")

larvae_gen = generations[1]
larvae_percent = percentages[1]

for (climate_group in cg_larvae_med_doy){
  for (future_version in future_versions){
    if(climate_group == "Historical") { future_version = "rcp85"}
    if(future_version == "rcp45") { data = d_rcp45} else { data = d }
    
    for (larvae_gen in generations){
      for (larvae_percent in percentages){
        genPct = paste0(type, larvae_gen, "_", larvae_percent)
        layerlist = levels(data$timeFrame)
        sub_Gen = subset(data, !is.na(timeFrame) & !is.na(get(genPct)) & timeFrame == climate_group, 
                         select = c(timeFrame, year, location, get(genPct)))

        sub_Gen = sub_Gen[, .(medianDoY = as.integer(median( get(genPct) ))), 
                           by = c("timeFrame", "location")]

        medianGen = list( hist = subset(sub_Gen, timeFrame == layerlist[1]),
                         `2040` = subset(sub_Gen, timeFrame == layerlist[2]),
                         `2060` = subset(sub_Gen, timeFrame == layerlist[3]),
                         `2080` = subset(sub_Gen, timeFrame == layerlist[4]))
        GenMap <- constructMap(medianGen, layerlist, palColumn = "medianDoY",
                               legendVals = domainVal, "Median Day of Year")
        GenMap
        saveWidget(GenMap, "temp.html", selfcontained = FALSE)
        file_name = paste0("Egg_Hatch_MDoY_", climate_group, "_", 
                           future_version, "_", larvae_gen, "_", 
                           larvae_percent,".pdf")
        webshot("temp.html", file = file_name, cliprect = "viewport")

      }
    }
  }
}
################
############################ Difference From Historical
################
type_diff = "L"
diffType = 3
cg_larvae_doy_diffS = c("2040's - Historical_rcp45", 
                        "2040's - Historical_rcp48",
                       
                        "2060's - Historical_rcp45", 
                        "2060's - Historical_rcp85", 
                       
                        "2080's - Historical_rcp45",
                        "2080's - Historical_rcp85")

generations = c("Gen1", "Gen2", "Gen3", "Gen4")
percentages = c("0.25", "0.5", "0.75")

cg_larvae_doy_diff = cg_larvae_doy_diffS[1]
larvae_gen_diff = generations[1]
larvae_percent_diff = percentages[1]

for (cg_larvae_doy_diff in cg_larvae_doy_diffS){
  for (larvae_gen_diff in generations){
    for (larvae_percent_diff in percentages){
      genPct = paste0(type_diff, larvae_gen_diff, "_", larvae_percent_diff)
      temp = tstrsplit(cg_larvae_doy_diff, "_")
      climate_group = unlist(temp[1])
      future_version = unlist(temp[2])
      if(future_version == 'rcp45') {data = d_rcp45} else { data = d }
      layerdiff = c("2040's - Historical", "2060's - Historical", "2080's - Historical")
      layerlist = levels(data$timeFrame)

      sub_Gen = subset(data, !is.na(timeFrame) & !is.na(get(genPct)), 
                       select = c(timeFrame, year, location, get(genPct)))
      sub_Gen = sub_Gen[, .(value = as.integer(quantile( get(genPct), names = FALSE )[diffType])), 
                           by = c("timeFrame", "location")]
          
      tfGen = list(subset(sub_Gen, timeFrame == layerlist[1]),
                   subset(sub_Gen, timeFrame == layerlist[2]),
                   subset(sub_Gen, timeFrame == layerlist[3]),
                   subset(sub_Gen, timeFrame == layerlist[4])
                   )
      if(layerdiff[1] == climate_group) {
        diffGen = list(merge(tfGen[[2]], tfGen[[1]], by = c("location")))
          } else if(layerdiff[2] == climate_group) {
            diffGen = list(merge(tfGen[[3]], tfGen[[1]], by = c("location")))
          } else if(layerdiff[3] == climate_group) {
            diffGen = list(merge(tfGen[[4]], tfGen[[1]], by = c("location")))
          }
      diffGen[[1]]$diff = diffGen[[1]]$value.y - diffGen[[1]]$value.x
      diffDomain = diffGen[[1]]$diff
              
      GenDiffMap <- constructMap(diffGen, 
                                 layerdiff, 
                                 palColumn = "diff", 
                                 legendVals = seq(40, 140), 
                                 HTML("Median calendar day<br />difference from historical"), 
                                 RdBu_reverse)
      # GenDiffMap
      saveWidget(GenDiffMap, "temp.html", selfcontained = FALSE)
      file_name = paste0("Egg_Hatch_diffH_", 
                         cg_larvae_doy_diff, "_", 
                         larvae_gen_diff, "_", 
                         larvae_percent_diff,".pdf")
      webshot("temp.html", file = file_name, cliprect = "viewport")

    }
  }
}

################################################################################################
################
################                      CM Flight
################
################################################################################################
rm(list=ls())
library(shiny)
library(shinydashboard)
library(htmlwidgets)
library(webshot)
library(shinyBS)
library(rgdal)    # for readOGR and others
library(maps)
library(sp)       # for spatial objects
library(leaflet)  # for interactive maps (NOT leafletR here)
library(dplyr)    # for working with data frames
library(ggplot2)  # for plotting
library(data.table)
library(reshape2)
library(RColorBrewer)

source_path = "/Users/hn/Documents/GitHub/Kirti/codling_moth/code/laptop_codes/map_functions.R"
source(source_path)

# data_dir <- "/Users/hn/Desktop/Kirti/check_point/CAHNR/"
data_dir <- "/Users/hn/Desktop/Kirti/check_point/CAHNR/2015/"
setwd(data_dir)

########### Giridhar Data 
# d = data.table(readRDS(paste0(data_dir,"combinedData.rds")))
# d$timeFrame <-as.factor(d$timeFrame)
# d$timeFrame <- factor(d$timeFrame, levels = levels(d$timeFrame)[c(4,1,2,3)])

# d_rcp45 = data.table(readRDS(paste0(data_dir,"combinedData_rcp45.rds")))
# names(d_rcp45)[names(d_rcp45) == "ClimateGroup"] = "timeFrame"
# d_rcp45$location = paste0(d_rcp45$latitude, "_", d_rcp45$longitude)
# cols <- colnames(d)
# d_rcp45 <- subset(d_rcp45, select=cols)

d = data.table(readRDS(paste0(data_dir,"combinedData.rds")))
d$timeFrame <-as.factor(d$timeFrame)
d$timeFrame <- factor(d$timeFrame, levels = levels(d$timeFrame)[c(4,1,2,3)])

d_rcp45 = data.table(readRDS(paste0(data_dir,"combinedData_rcp45.rds")))

RdBu_reverse <- rev(brewer.pal(11, "RdBu"))
#######################################
####################################### Median Day Of Year (First Flight)
#######################################

time_period = c("Historical", "2040's", "2060's", "2080's")
model_types = c("rcp45", "rcp85")
col = "Emergence"
for (climate_group in time_period) {
  for (future_version in model_types) {
    if (climate_group == "Historical") {future_version = "rcp85"}
    if(future_version == "rcp45") {data = d_rcp45} else { data = d}
    layerlist = levels(data$timeFrame) # c("Historical", "2040's", "2060's", "2080's")
    sub_Emerg = subset(data, !is.na(timeFrame) & !is.na(get(col)) & 
                       timeFrame == climate_group, 
                       select = c(timeFrame, year, location, get(col))
                       )
    sub_Emerg = sub_Emerg[, .(medianDoY = as.integer(median( get(col) ))), by = c("timeFrame", "location")]
    medianEmerg = list(hist = subset(sub_Emerg, timeFrame == layerlist[1]),
                       `2040` = subset(sub_Emerg, timeFrame == layerlist[2]),
                       `2060` = subset(sub_Emerg, timeFrame == layerlist[3]),
                       `2080` = subset(sub_Emerg, timeFrame == layerlist[4]))
    EmergMap <- constructMap(medianEmerg, layerlist, 
                             palColumn = "medianDoY", 
                             legendVals = seq(65,145), "Median Day of Year")
    EmergMap
    saveWidget(EmergMap, "temp.html", selfcontained = FALSE)
    file_name = paste0("CM_Flight_Median_DoY_", 
                        climate_group, "_", 
                        future_version,".pdf")
    webshot("temp.html", file = file_name, cliprect = "viewport")
  }
}
#######################################
####################################### Difference from Historical (First Flight)
#######################################
time_period = c("2040's - Historical", "2060's - Historical", "2080's - Historical")
model_types = c("rcp45", "rcp85")
diffType = 3
col = "Emergence"

for (climate_group in time_period){
  for (future_version in model_types){
    if(future_version == "rcp45") { data = d_rcp45} else { data = d }
    layerdiff = c("2040's - Historical", "2060's - Historical", "2080's - Historical")
    layerlist = levels(data$timeFrame)
    sub_Emerg = subset(data, !is.na(timeFrame) & !is.na(get(col)), 
                       select = c(timeFrame, year, location, get(col)))
    sub_Emerg = sub_Emerg[, .(value = as.integer(quantile( get(col), names = FALSE )[diffType])), 
                           by = c("timeFrame", "location")]
    tfEmerg = list(subset(sub_Emerg, timeFrame == layerlist[1]),
                   subset(sub_Emerg, timeFrame == layerlist[2]),
                   subset(sub_Emerg, timeFrame == layerlist[3]),
                   subset(sub_Emerg, timeFrame == layerlist[4]))
    if(layerdiff[1] == climate_group) {
      diffEmerg = list(merge(tfEmerg[[2]], tfEmerg[[1]], by = c("location")))
    }
    else if(layerdiff[2] == climate_group) {
      diffEmerg = list(merge(tfEmerg[[3]], tfEmerg[[1]], by = c("location")))
    }
    else if(layerdiff[3] == climate_group) {
      diffEmerg = list(merge(tfEmerg[[4]], tfEmerg[[1]], by = c("location")))
    }
    diffEmerg[[1]]$diff = diffEmerg[[1]]$value.y - diffEmerg[[1]]$value.x
    diffDomain = diffEmerg[[1]]$diff

    EmergDiffMap <- constructMap(diffEmerg, layerdiff, 
                                 palColumn = "diff", 
                                 legendVals = seq(0,45), 
                                 HTML("Median calendar day<br />difference from historical"), 
                                 RdBu_reverse)
    saveWidget(EmergDiffMap, "temp.html", selfcontained = FALSE)
    file_name = paste0("CM_Flight_Diff_H_1st_Flight_", 
                       climate_group, "_", 
                       future_version, ".pdf")
    webshot("temp.html", file = file_name, cliprect = "viewport")
  }
}

#######################################
####################################### Median Day of Year (By Generation)
#######################################
time_period = c("Historical", "2040's", "2060's", "2080's")
model_types = c("rcp45", "rcp85")
generations = c("Gen1", "Gen2", "Gen3", "Gen4")
percentages = c("0.25", "0.5", "0.75")

type = "A"
domainVal = seq(85, 365)

for (climate_group in time_period){
  for (future_version in model_types){
    if (climate_group == "Historical") {future_version = "rcp85"}
    if (future_version == "rcp85"){data = d} else {data = d_rcp45}
    for (adult_gen in generations){
      for(adult_percent in percentages){
        genPct = paste0(type, adult_gen, "_", adult_percent)
        layerlist = levels(data$timeFrame) 
        sub_Gen = subset(data, !is.na(timeFrame) & !is.na(get(genPct)) & 
                         timeFrame == climate_group, 
                         select = c(timeFrame, year, location, 
                          get(genPct)))
        
        sub_Gen = sub_Gen[, .(medianDoY = as.integer(median( get(genPct) ))), 
                          by = c("timeFrame", "location")]

        medianGen = list( hist = subset(sub_Gen, timeFrame == layerlist[1]),
                         `2040` = subset(sub_Gen, timeFrame == layerlist[2]),
                         `2060` = subset(sub_Gen, timeFrame == layerlist[3]),
                         `2080` = subset(sub_Gen, timeFrame == layerlist[4]))
        GenMap <- constructMap(medianGen, layerlist, palColumn = "medianDoY", 
                               legendVals = domainVal, "Median Day of Year")
        saveWidget(GenMap, "temp.html", selfcontained = FALSE)
        file_name = paste0("CM_Flight_MDoY_by_Gen_", 
                           climate_group, "_", 
                           future_version, "_",
                           adult_gen, "_",
                           adult_percent, ".pdf")
        webshot("temp.html", file = file_name, cliprect = "viewport")
      }
    }
  }
}
#######################################
####################################### Difference from Historical (by Generations)
#######################################
time_period = c("2040's - Historical", "2060's - Historical", "2080's - Historical")
model_types = c("rcp45", "rcp85")
generations = c("Gen1", "Gen2", "Gen3", "Gen4")
percentages = c("0.25", "0.5", "0.75")

type_diff = "A"
diffType = 3

for (climate_group in time_period){
  for (future_version in model_types){
    if(future_version == 'rcp45') { data = d_rcp45} else { data = d }
    for (adult_gen_diff in generations){
      for (adult_percent_diff in percentages){
        genPct = paste0(type_diff, adult_gen_diff, "_", adult_percent_diff)
        layerdiff = c("2040's - Historical", "2060's - Historical", "2080's - Historical")
        layerlist = levels(data$timeFrame)
        
        sub_Gen = subset(data, !is.na(timeFrame) & !is.na(get(genPct)), 
                         select = c(timeFrame, year, location, get(genPct)))
        
        sub_Gen = sub_Gen[, .(value = as.integer(quantile( get(genPct), names = FALSE )[diffType])), 
                            by = c("timeFrame", "location")]

        tfGen = list(subset(sub_Gen, timeFrame == layerlist[1]),
                     subset(sub_Gen, timeFrame == layerlist[2]),
                     subset(sub_Gen, timeFrame == layerlist[3]),
                     subset(sub_Gen, timeFrame == layerlist[4]))

        if(layerdiff[1] == climate_group) {
          diffGen = list(merge(tfGen[[2]], tfGen[[1]], by = c("location")))
        } else if(layerdiff[2] == climate_group) {
          diffGen = list(merge(tfGen[[3]], tfGen[[1]], by = c("location")))
        } else if(layerdiff[3] == climate_group) {
          diffGen = list(merge(tfGen[[4]], tfGen[[1]], by = c("location")))
        }
        diffGen[[1]]$diff = diffGen[[1]]$value.y - diffGen[[1]]$value.x
        diffDomain = diffGen[[1]]$diff

        GenDiffMap <- constructMap(diffGen, layerdiff, 
                                   palColumn = "diff", 
                                   legendVals = seq(0,115), 
                                   HTML("Median calendar day<br />difference from historical"), 
                                   RdBu_reverse)
        saveWidget(GenDiffMap, "temp.html", selfcontained = FALSE)
        file_name = paste0("CM_Flight_Diff_by_Gen_", 
                           climate_group, "_", 
                           future_version, "_",
                           adult_gen_diff, "_",
                           adult_percent_diff, ".pdf")
        webshot("temp.html", file = file_name, cliprect = "viewport", zoom = 1)
      }
    }
  }
}
