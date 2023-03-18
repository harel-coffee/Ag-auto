# We could/should create two sets of data for each 
# (of the first two) 
# scenarios above or, we can take care of NAs
# -introduced to data by merging frost and bloom-
# in the plotting functions?
#####################################
rm(list=ls())
.libPaths("/data/hydro/R_libs35")
.libPaths()
library(data.table)
library(dplyr)
library(ggpubr)

options(digits=9)
options(digit=9)
############################################################
###
###             local computer source
###
############################################################
basee <- "/Users/hn/Documents/00_GitHub/Ag/Bloom/"
source_1 = paste0(basee, "bloom_core.R")
source_2 = paste0(basee, "/bloom_plot_core.R")
source(source_1)
source(source_2)

in_dir <- "/Users/hn/Documents/01_research_data/bloom/"
param_dir <- paste0(basee, "parameters/")
frost_plot_dir <- "/Users/hn/Documents/00_GitHub/Ag_papers/Chill_Paper/figures/first_frost/"
if (dir.exists(frost_plot_dir) == F) {
  dir.create(path = frost_plot_dir, recursive = T)
}
#############################################################
###
###               Read data off the disk
###
#############################################################
#
# for sake of iteration
#
location_ls <- read.csv(file = paste0(param_dir, "limited_locations.csv"), 
                        header=TRUE, as.is=TRUE)
location_ls$location <- paste0(location_ls$lat, "_", location_ls$long)
location_ls <- within(location_ls, remove(lat, long))
#
# convert chill day of year to readable day of year!
#
chill_doy_map <- read.csv(paste0(param_dir, "/chill_DoY_map.csv"), 
                          as.is=TRUE)
#############################################################

frost <- readRDS(paste0(in_dir, "first_frost.rds"))
frost <- frost %>%
         filter(location %in% location_ls$location) %>%
         data.table()

frost <- dplyr::left_join(x = frost, y = location_ls, by = "location")

ict <- c("Omak", "Yakima", "Walla Walla", "Eugene")

frost <- frost %>% 
         filter(city %in% ict) %>% 
         data.table()

frost$city <- factor(frost$city, levels = ict, order=TRUE)


#############################################################
#
# pick up 2026-2099 time period
#
#############################################################

frost <- frost %>% 
         filter(chill_season >= "chill_2025-2026")%>% 
         data.table()
#############################################################
#
#              clean up each data table
#
#############################################################
frost <- within(frost, remove(tmin, location, lat, long))
print(length(unique(frost$chill_season)))
print(length(unique(frost$model)))
#############################################################
emissions <- c("RCP 4.5", "RCP 8.5")

###############################################################
em <- emissions[2]
ct <- location_ls$city[1]
setnames(frost, old=c("city"), new=c("location"))
start_time <- Sys.time()


for (em in emissions){
  curr_frost <- frost %>% 
                filter(emission==em) %>% 
                data.table()

  #############################################################
  ##
  ##             Frost plot here
  ##
  source(source_2)
  frost_plt <- cloudy_frost_2_rows(d1 = curr_frost, 
                                   colname = "chill_dayofyear", 
                                   fil="first frost")

  ggsave(plot = frost_plt,
         filename = paste0("first_frost_", gsub(" ", "", gsub("\\.", "", em)), ".png"), 
         width=15, height=10, units = "in", 
         dpi=400, device = "png",
         path=frost_plot_dir)
}

print(Sys.time() - start_time)


