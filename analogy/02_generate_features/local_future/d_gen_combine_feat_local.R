.libPaths("/data/hydro/R_libs35")
.libPaths()
library(data.table)
library(dplyr)
library(geepack)
library(chron)

source_path = "/home/hnoorazar/analog_codes/core_analog.R"
source(source_path)

options(digits=9)
options(digit=9)

param_dir = "/home/hnoorazar/cleaner_codes/parameters/"
main_in_dir <- "/data/hydro/users/Hossein/analog/local/data_bases/"
main_out_dir <- "/data/hydro/users/Hossein/analog/local/ready_features/"

##################################################################
##
##            Terminal Arguments
##
##################################################################

##################################################################
carbon_types = c("rcp45", "rcp85")

for (carbon_type in carbon_types){

  #### Create First Flight Median Day of Year
  FF_dt <- data.table(readRDS(paste0(main_in_dir, "/001_unique_CM/", "short_combined_CM_", carbon_type, ".rds")))
  FF_dt <- generate_mDoY_FF(FF_dt, meann=TRUE) # meann is for taking the mean across models
  
  ##
  ## Create No. of Generations. We only want Larva by Aug 23.
  ## There are 19 models, hence, we have to take averages of number of generations
  ##
  gen_dt <- data.table(readRDS(paste0(main_in_dir, "generations_", carbon_type, ".rds")))
  gen_dt <- subset(gen_dt, select = c("year", "location", "NumLarvaGens_Aug"))

  gen_dt <- gen_dt %>% 
            group_by(location, year) %>% 
            summarise_at(vars(NumLarvaGens_Aug), funs(mean(., na.rm=TRUE)))%>% 
            data.table()
  ###
  ### Create relative fraction of escaped diapause
  ###
  diapause_dt <- data.table(readRDS(paste0(main_in_dir, "diapause_map1_clean_", carbon_type, ".rds")))
  diapause_dt <- subset(diapause_dt, select=c("location", "year", 
                                              "RelPctNonDiapGen1", "RelPctNonDiapGen2",
                                              "RelPctNonDiapGen3", "RelPctNonDiapGen4"))

  diapause_dt <- diapause_dt %>%
                 group_by(location, year) %>%
                 summarise_at(.funs = funs(mean(., na.rm=TRUE)), 
                              vars(RelPctNonDiapGen1 : RelPctNonDiapGen4))%>% 
                 data.table()
  
  setnames(diapause_dt, old=c("RelPctNonDiapGen1", "RelPctNonDiapGen2", 
                              "RelPctNonDiapGen3", "RelPctNonDiapGen4"), 
                        new=c("mean_escaped_Gen1", "mean_escaped_Gen2",
                              "mean_escaped_Gen3", "mean_escaped_Gen4"))

  ### Read precip
  precip_dt <- data.table(readRDS(paste0(main_in_dir, "precip_", carbon_type, ".rds")))
  precip_dt <- precip_dt[, .(mean_precip = mean(precip)), by = c("location", "year")]

  ### Read gdd
  gdd_dt <- data.table(readRDS(paste0(main_in_dir, "gdd_", carbon_type, ".rds")))
  # mean over models/ClimateScenario
  gdd_dt <- gdd_dt[, .(mean_gdd = mean(CumDDinF)), by = c("location", "year")]

  #
  # merge several data frames:
  #
  all_data_dt <- Reduce(function(...) merge(..., all = T), 
                        list(FF_dt, gen_dt, diapause_dt, precip_dt, gdd_dt))
  
  # there are NA values in escaped population of 4th generations!
  # I have to look into it to see why, but perhaps the reason is
  # that there is no such a thing. So, for now, I replace them with zeros
  all_data_dt$mean_escaped_Gen4[is.na(all_data_dt$mean_escaped_Gen4)] <- 0
  all_data_dt$mean_escaped_Gen3[is.na(all_data_dt$mean_escaped_Gen3)] <- 0
  all_data_dt$treatment <- 1
  all_data_dt$ClimateScenario <- "ensembe_mean"

  saveRDS(all_data_dt, paste0(main_out_dir, "averaged_data/", "averaged_data_", carbon_type, ".rds"))

  write.csv(all_data_dt, 
            paste0(main_out_dir, "averaged_data/", "averaged_data_", carbon_type, ".csv"),
            row.names=FALSE)
}


