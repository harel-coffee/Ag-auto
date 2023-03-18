# #!/share/apps/R-3.2.2_gcc/bin/Rscript  # old from 2018


#######!/share/apps/R-3.2.2_gcc/bin/Rscript

.libPaths("/data/hydro/R_libs35")
.libPaths()

library(data.table)
library(lubridate)
library(reshape2)
library(dplyr)
library(foreach)
library(iterators)
library(chron)
options(digits=9)

source_path = "/home/hnoorazar/codling_moth_2021/core.R"
source(source_path)

dir_base = "/data/hydro/users/Hossein/codling_moth_2021/"
param_dir = "/home/hnoorazar/codling_moth_2021/parameters/"
write_dir = paste0(dir_base, "01_combined_files/")

locations = data.table(read.csv(paste0(param_dir, "five_locations.csv")))
locations = locations$location

args = commandArgs(trailingOnly=TRUE)
sub_dir <- args[1]
emission <- args[2] # emission should be observed, historical, rcp45, rcp85

if (emission == "observed"){
  models = c("observed")
   } else {
  models = c("bcc-csm1-1-m", "BNU-ESM", "CanESM2", "CNRM-CM5", "GFDL-ESM2G",  "GFDL-ESM2M")
}

file_pref <- tail(strsplit(sub_dir, "_")[[1]], 1)

# input_dir, locations, models, file_prefix, emission
data <- merge_data(input_dir = paste0(dir_base, sub_dir, "/"), 
                   locations = locations,
                   models = models,
                   file_prefix = file_pref,
                   emission = emission)

if (emission == "observed"){
    em <- "Observed"
  } else if (emission == "rcp45"){
    em <- "RCP 4.5"
  } else if (emission == "rcp85"){
    em <- "RCP 8.5"
  } else if (emission == "historical"){
    em <- "Historical"
}

data$emission <- em

locations <- data.table(read.csv(paste0(param_dir, "five_locations.csv")))
locations <- within(locations, remove(lat, long))

data$location <-  paste0(data$latitude, "_", data$longitude)
data <- merge(data, locations, by="location", all.x=T)

L = c("East Oroville", "Wenatchee", "Quincy", "Wapato", "Richland")
data$city <- factor(data$city, levels = L, ordered = TRUE)

L = c("1979-2015", "2040s", "2060s", "2080s")
data$time_period <- factor(data$time_period, levels = L, ordered = TRUE)

#
# strsplit("sub_dir", "_")[[1]][2] gives LO or LM and in case we need allUSA later!
#
dir.create(file.path(write_dir), recursive = TRUE)
saveRDS(object=data, 
        file = paste0(write_dir,  strsplit(sub_dir, "_")[[1]][2], "_combined_", file_pref, "_", emission, ".rds"))

# write.csv(data, paste0(write_dir, strsplit(sub_dir, "_")[[1]][2], "_combined_", file_pref, "_", emission, ".csv"), row.names = F)





