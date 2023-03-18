#!/share/apps/R-3.2.2_gcc/bin/Rscript

library(data.table)
library(dplyr)
library(foreach)

read_dir <- "/data/hydro/users/Hossein/chill/7_time_intervals/"
write_dir<- read_dir

param_dir <- "/home/hnoorazar/chilling_codes/parameters/"
limited_locations <- read.csv(paste0(param_dir, "limited_locations.csv"), header=T, sep=",", as.is=T)

file_names = paste0(limited_locations$lat, "_", limited_locations$long)
city_names <- limited_locations$city

print (file_names)



# city_names = c("Omak", "Richland", "Wenatchee", "Hilsboro", "Elmira")

climate_scenarios <- list.files(read_dir, all.files=F, include.dirs=F)
climate_scenarios <- c("BNU-ESM", "CCSM4", "CNRM-CM5", "CSIRO-Mk3-6-0",
                       "CanESM2", "GFDL-ESM2G", "GFDL-ESM2M", "HadGEM2-CC365",
                       "HadGEM2-ES365",  "IPSL-CM5A-LR", "IPSL-CM5A-MR", "IPSL-CM5B-LR",
                       "MIROC-ESM-CHEM", "MIROC5", "MRI-CGCM3", "NorESM1-M",
                       "bcc-csm1-1", "bcc-csm1-1-m", "inmcm4", "observed")

print ("__________________________________")
print ("climate_scenarios are ")
print (climate_scenarios)
print ("__________________________________")
projection_type = c("historical", "rcp45", "rcp85")

observed = data.table()
modeled_hist = data.table()
rcp45 = data.table()
rcp85 = data.table()

start_time <- Sys.time()

for (cs in climate_scenarios){
    if (cs == "observed"){
  curr_path = file.path(read_dir, cs)
  for (count in seq(1, nrow(limited_locations))){
      f = data.table(readRDS(paste0(curr_path, "/met_hourly_data_", file_names[count], ".rds")))
      f$climateScenario = "observed"
      f$location = file_names[count]
      f$city = city_names[count]
      observed <- rbind(observed, f)
  }
    } else {
  for (proj in projection_type){
     curr_path = file.path(read_dir, cs, proj)

      for (count in seq(1, nrow(limited_locations))){
    f = data.table(readRDS(paste0(curr_path, "/met_hourly_data_", file_names[count], ".rds")))
    f$climateScenario = cs
    f$location = file_names[count]
    f$city = city_names[count]

     if (proj=="historical"){
           modeled_hist <- rbind(modeled_hist, f)
          } else if (proj=="rcp45"){
        rcp45 <- rbind(rcp45, f)
        } else {
         rcp85 <- rbind(rcp85, f)
    }
      }
  }
    }
}
# pick up the variables we need
needed_cols = c("year", "month", "Temp", "chill_season", "climateScenario", "location", "city")

observed = subset(observed, select=needed_cols)
rcp45 = subset(rcp45, select=needed_cols)
rcp85 = subset(rcp85, select=needed_cols)
modeled_hist = subset(modeled_hist, select=needed_cols)

## pick up the months we are interested in:
# to make the files smaller
# Months of Interest:
mos = c(9, 10, 11, 12, 1, 2, 3)
observed <- observed  %>% filter(month %in% mos)
rcp45 <- rcp45  %>% filter(month %in% mos)
rcp85 <- rcp85  %>% filter(month %in% mos)
modeled_hist <- modeled_hist  %>% filter(month %in% mos)

observed$scenario = "observed"
colnames(observed)[colnames(observed) == 'climateScenario'] <- 'model'
saveRDS(observed, paste0(write_dir, "observed.rds"))

colnames(modeled_hist)[colnames(modeled_hist) == 'climateScenario'] <- 'model'
colnames(rcp45)[colnames(rcp45) == 'climateScenario'] <- 'model'
colnames(rcp85)[colnames(rcp85) == 'climateScenario'] <- 'model'

modeled_hist$scenario = "historical"
rcp45$scenario = "rcp45"
rcp85$scenario = "rcp85"

saveRDS(modeled_hist, paste0(write_dir, "modeled_hist.rds"))
saveRDS(rcp45, paste0(write_dir, "rcp45.rds"))
saveRDS(rcp85, paste0(write_dir, "rcp85.rds"))

modeled = bind_rows(modeled_hist, rcp45, rcp85)
saveRDS(modeled, paste0(write_dir, "modeled.rds"))
rm(rcp85, rcp45, modeled_hist)

mos = c(9, 10, 11, 12, 1, 2, 3)
month_names = c("Jan", "Feb", "Mar", "Apr" , "May", "Jun", "Jul", "Aug", "Sept", "Oct", "Nov", "Dec")

for (montH in mos){
    data <- modeled %>% filter(month == montH)
    saveRDS(data, paste0(write_dir, month_names[montH], ".rds"))
    rm(data)
    data <- observed %>% filter(month == montH)
    saveRDS(data, paste0(write_dir, "observed_", month_names[montH], ".rds"))
}
########################################################
#######                                          #######
#######     Sept-through-Apr (excludes April)    #######
#######                                          #######
########################################################
mos = c(9, 10, 11, 12, 1, 2, 3)
modeled <- modeled  %>% filter(month %in% mos)
saveRDS(modeled, paste0(write_dir, "sept_thru_Apr_modeled.rds"))

observed <- observed  %>% filter(month %in% mos)
saveRDS(observed, paste0(write_dir, "sept_thru_Apr_observed.rds"))


print( Sys.time() - start_time)


