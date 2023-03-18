######################################################################
rm(list=ls())
library(lubridate)
library(ggpubr)
library(purrr)
library(tidyverse)
library(data.table)
library(dplyr)
library(ggplot2)
library(maps)

options(digit=9)
options(digits=9)

##########################################################################
####
####         functions here
####
##########################################################################
plot_the_map <- function(a_dt, county2, title_p){

  curr_plot <- ggplot(a_dt, aes(long, lat, group = group)) + 
               geom_polygon(data = county2, fill="lightgrey") +
               geom_polygon(aes(fill = analog_freq), colour = rgb(1, 1, 1, 0.2))  +
               coord_quickmap() + 
               theme(legend.title = element_blank(),
                     axis.text.x = element_blank(),
                     axis.text.y = element_blank(),
                     axis.ticks.x = element_blank(),
                     axis.ticks.y = element_blank(),
                     axis.title.x = element_blank(),
                     axis.title.y = element_blank()) + 
               ggtitle(title_p)
   return(curr_plot) 
}
######################################################################
####
####         Set up directories
####
######################################################################

data_sub_dirs <- c("no_no_85/", "no_w_85/", "w_no_85/", "w_w_85/", 
                   "no_no_45/", "no_w_45/", "w_no_45/", "w_w_45/")

data_dir <- paste0("/Users/hn/Desktop/Desktop/Kirti/check_point/analogs/", data_sub_dirs[1])
param_dir <- "/Users/hn/Documents/GitHub/Kirti/analogy/parameters/"

######################################################################
####
####           global Files
####
######################################################################
local_cnty_fips <- "local_county_fips.csv"
usa_cnty_fips <- "all_us_1300_county_fips_locations.csv"
local_fip_cnty_name_map <- "17_counties_fips_unique.csv"

local_cnty_fips <- data.table(read.csv(paste0(param_dir, local_cnty_fips), header=T, sep=",", as.is=T))
usa_cnty_fips <- data.table(read.csv(paste0(param_dir, usa_cnty_fips), header=T, sep=",", as.is=T))
local_fip_cnty_name_map <- data.table(read.csv(paste0(param_dir, local_fip_cnty_name_map), 
                                               header=T, sep=",", as.is=T))

local_cnty_fips <- local_cnty_fips %>% filter(location %in% usa_cnty_fips$location)
local_fips <- unique(local_cnty_fips$fips)

model_names <- c("bcc-csm1-1-m", "BNU-ESM", "CanESM2", "CNRM-CM5", "GFDL-ESM2G", "GFDL-ESM2M")
time_periods <- c("2026_2050", "2051_2075", "2076_2095")
emissions <- c("rcp45", "rcp85")

target_fip <- local_fips[1]
model_n <- model_names[1]
time_p <- time_periods[1]
emission <- emissions[2]

for (time_p in time_periods){
  for (target_fip in local_fips){
    for(model_n in model_names){
      analog_file_name <- paste("analog", model_n, emission, time_p, sep="_")
      analog_dat <- data.table(readRDS(paste0(data_dir, analog_file_name, ".rds")))

      # replace no analogs with na to be able to omit them
      # analog_dat$analog_NNs_county[analog_dat$analog_NNs_county == "no_analog"] <- NA

      analog_dat <- analog_dat %>% filter(analog_NNs_county != "no_analog")
      analog_dat$analog_NNs_county <- as.integer(analog_dat$analog_NNs_county)
      analog_dat$query_county <- as.integer(analog_dat$query_county)

      # analog_dat <- na.omit(analog_dat)
      analog_dat <- analog_dat %>% filter(query_county == target_fip)

      one_mod_map_info <- produce_dt_for_map(analog_dat)

      data(county.fips) # Load the county.fips dataset for plotting
      cnty <- map_data("county") # Load the county data from the maps package
      cnty2 <- cnty %>%
               mutate(polyname = paste(region, subregion, sep=",")) %>%
               left_join(county.fips, by="polyname")

      # one_mod_map_info <- left_join(analog_dat, cnty2, by=c("analog_NNs_county" = "fips"))

      target_cnty_name <- local_fip_cnty_name_map$st_county[local_fip_cnty_name_map$fips==target_fip]
      target_cnty_name <- paste(unlist(strsplit(target_cnty_name, "_"))[2], 
                                unlist(strsplit(target_cnty_name, "_"))[1], sep= ", ")
      the_title <- paste0(target_cnty_name, " (", 
                       paste(unlist(strsplit(time_p, "_"))[1], 
                             unlist(strsplit(time_p, "_"))[2], sep="-"),
                       ", ", model_n, emission, ")" )

      assign(x = gsub("-", "_", model_n), 
             value = {plot_the_map(one_mod_map_info, cnty2, the_title)})
      
    }
    assign(x = paste0("map_", target_fip) , 
           value={ggarrange(plotlist = list(bcc_csm1_1_m, BNU_ESM, CanESM2,
                                            CNRM_CM5, GFDL_ESM2G, GFDL_ESM2M),
                            ncol = 6, nrow = 1, common.legend = TRUE)})
  }
  assign(x = paste0("map_", time_p) , 
         value={ggarrange(plotlist = list(map_16027, map_41021, map_41027, 
                                          map_41049, map_41059, map_53001, 
                                          map_53005, map_53007, map_53013, 
                                          map_53017, map_53021, map_53025, 
                                          map_53037, map_53039, map_53047, 
                                          map_53071, map_53077),
                          ncol = 1, nrow = 17, common.legend = TRUE)})
}


master_path <- paste0(data_dir, "/maps/")
if (dir.exists(master_path) == F) { dir.create(path = master_path, recursive = T)}

ggsave("map_2026_2050.png", map_2026_2050, 
       path=master_path, device="png",
       dpi=300, width=40, height=100, unit="in", limitsize = FALSE)

ggsave("map_2051_2075.png", map_2051_2075, 
       path=master_path, device="png",
       dpi=300, width=40, height=100, unit="in", limitsize = FALSE)

ggsave("map_2076_2095.png", map_2076_2095, 
       path=master_path, device="png",
       dpi=300, width=40, height=100, unit="in", limitsize = FALSE)



