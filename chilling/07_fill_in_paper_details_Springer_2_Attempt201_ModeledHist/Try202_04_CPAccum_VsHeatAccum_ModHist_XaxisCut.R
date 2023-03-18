##
##   Jan 31, 2022. 
##   This is copy of Try201_04_CPAccum_VsHeatAccum_ModHist.R
##   with one edit: cut the y-axis at 400.
##
##
rm(list=ls())

library(data.table)
library(dplyr)
library(ggpubr)
library(ggplot2)

options(digits=9)
options(digit=9)
start_time <- Sys.time()
############################################################
###
###             local computer source
###
############################################################
bloom_source_dir <- "/Users/hn/Documents/00_GitHub/Ag/Bloom/"
chill_source_dir <- "/Users/hn/Documents/00_GitHub/Ag/chilling/"

in_dir <- "/Users/hn/Documents/01_research_data/bloom/"
param_dir <- paste0(bloom_source_dir, "parameters/")

plot_base_dir <- "/Users/hn/Documents/00_GitHub/ag_papers/chill_paper/02_Springer_2/figure_202_ModHist/"

daily_CP_dir <- "/Users/hn/Documents/01_research_data/chilling/01_data/"
daily_vertDD_dir <- "/Users/hn/Documents/01_research_data/bloom_4_chill_paper_trigger/"
#############################################################
###
### 
###
#############################################################

bloom_core_source <- paste0(bloom_source_dir, "bloom_core.R")
bloom_plot_core_source <- paste0(bloom_source_dir, "bloom_plot_core.R")

chill_core_source <- paste0(chill_source_dir, "chill_core.R")

source(bloom_core_source)
source(bloom_plot_core_source)
source(chill_core_source)
#############################################################
###
###               Read data off the disk
###
#############################################################

########
######## read parameters
########

limited_locations <- read.csv(file = paste0(param_dir, "limited_locations.csv"), 
                              header=TRUE, as.is=TRUE)

limited_locations$location <- paste0(limited_locations$lat, "_", limited_locations$long)
limited_locations <- within(limited_locations, remove(lat, long))

chill_doy_map <- read.csv(paste0(param_dir, "/chill_DoY_map.csv"), as.is=TRUE)

########
######## read data
########
#__________   CP
daily_CP <- read.csv(paste0(daily_CP_dir, "daily_CP_for_CDF.csv"), as.is=TRUE)

daily_CP <- daily_CP %>% 
            filter(emission != "observed") %>% 
            data.table()

#__________   Heat

#
#    The code 01_copy_4_locations_from_01_step.R merges some files related
#    to the 4 locations we want. Its output name is bloom_01Step_4locations_for_chill.rds
#    Before, I skipped the modeled historical. Now, Oct. 11, 2021 I added those in.
#    After downloading the file to local machine, I renamed it to bloom_01Step_4locations_for_chill_wModHist.rds
#    
#
heatAccum <- readRDS(paste0(daily_vertDD_dir, "bloom_01Step_4locations_for_chill_wModHist.rds"))

heatAccum <- heatAccum %>% filter(model != "Observed") %>% data.table()

heatAccum$location <- paste0(heatAccum$lat, "_", heatAccum$long)

heatAccum <- within(heatAccum, 
             remove(cripps_pink, gala, red_deli, vert_Cum_dd, vert_Cum_dd_F, lat, long))

#
# Just for choosing chill_seasons
#
# data_dir <- "/Users/hn/Documents/01_research_data/bloom_4_chill_paper_trigger/"
# trigger_dtA <- readRDS(paste0(data_dir, "heatTriggers_sept_summary_comp.rds"))
# chosen_chillSeasons <- unique(trigger_dtA$chill_season)

#__________________________________________________________________________________
#
#  Some of the chill seasons at are not trimmed. Toss them. Harmonize the data
#

heatAccum <- heatAccum %>% 
             filter(time_period %in% c("1950-2005", "2026-2050", "2051-2075", "2076-2099")) %>% 
             data.table()

unwanted_chill_seasons <- c("chill_1949-1950", "chill_2005-2006", "chill_2025-2026", "chill_2099-2100")

heatAccum <- heatAccum %>% filter(!(chill_season %in% unwanted_chill_seasons)) %>% data.table()

# heatAccum <- heatAccum %>% 
#              filter(chill_season %in% chosen_chillSeasons) %>% 
#              data.table()
#__________________________________________________________________________________
#
# add time periods to the data
#
heatAccum[heatAccum$time_period == "1950-2005", "time_period"] <- "modeled historical"


daily_CP_modHist <- daily_CP %>%
                    filter(emission == "modeled historical") %>%
                    data.table()

daily_CP_modHist$time_period <- "modeled historical"

daily_CP_F <- daily_CP %>%
              filter(emission != "modeled historical") %>%
              data.table()

# daily_CP_obs$time_period <- "Observed"
daily_CP_F <- add_time_periods_model(daily_CP_F)
daily_CP_F <- daily_CP_F %>% filter(time_period != "2006-2025") %>% data.table()

daily_CP <- rbind(daily_CP_modHist, daily_CP_F)

daily_CP <- daily_CP %>% filter(!(chill_season %in% unwanted_chill_seasons)) %>% data.table()

# daily_CP <- daily_CP %>% 
#             filter(chill_season %in% chosen_chillSeasons) %>% 
#             data.table()

#############################################################
#
# pick up observed and 2026-2099 time period
#
#############################################################

# heatAccum <- pick_obs_and_F(heatAccum)
# daily_CP <- pick_obs_and_F(daily_CP)

#############################################################
#
#              clean up and prepare each data table
#
#############################################################

heatAccum <- dplyr::left_join(x = heatAccum, 
                              y = limited_locations, 
                              by = "location")

x <- sapply(daily_CP$location, 
            function(x) strsplit(x, "_")[[1]], 
            USE.NAMES=FALSE)
lat = x[1, ]
long = x[2, ]

daily_CP$lat <- lat
daily_CP$long <- long

daily_CP <- add_chill_sept_DoY(daily_CP)

heatAccum <- within(heatAccum, remove("location"))
daily_CP <- within(daily_CP, remove("location", "lat", "long"))

#############################################################

emissions <- c("RCP 4.5", "RCP 8.5")
setnames(heatAccum, old=c("city"), new=c("location"))
setnames(daily_CP, old=c("city"), new=c("location"))

################################################

ict <- c("Omak", "Yakima", "Walla Walla", "Eugene")

heatAccum$location <- factor(heatAccum$location, levels = ict, order=TRUE)
daily_CP$location <-  factor(daily_CP$location,  levels = ict, order=TRUE)


daily_CP[daily_CP$emission == "modeled historical", "emission"] <- "Historical"

daily_CP$emission <- factor(daily_CP$emission,
                            levels = c("Historical", "RCP 4.5", "RCP 8.5"), 
                            order = TRUE)

heatAccum$emission <- factor(heatAccum$emission, 
                             levels = c("Historical", "RCP 4.5", "RCP 8.5"), 
                             order = TRUE)


heatAccum[heatAccum$time_period == "modeled historical", "time_period"] <- "Historical"
daily_CP[daily_CP$time_period == "modeled historical", "time_period"] <- "Historical"

tpo <- c("Historical", "2026-2050", "2051-2075", "2076-2099")
daily_CP$time_period <- factor(daily_CP$time_period, 
                               levels = tpo,
                               order = TRUE)

heatAccum$time_period <- factor(heatAccum$time_period, 
                                levels = tpo, 
                                order = TRUE)

####################################################################
#
# the followins are not observed. But I am sick of changing the name to modeled hist.
# 
#
heatAccum_obs <- heatAccum %>% filter(time_period == "Historical") %>% data.table()
daily_CP_obs <- daily_CP %>% filter(time_period == "Historical") %>% data.table()

heatAccum_obs45 <- heatAccum_obs
heatAccum_obs85 <- heatAccum_obs

daily_CP_obs45 <- daily_CP_obs
daily_CP_obs85 <- daily_CP_obs

heatAccum_obs45$emission <- "RCP 4.5"
heatAccum_obs85$emission <- "RCP 8.5"

daily_CP_obs45$emission <- "RCP 4.5"
daily_CP_obs85$emission <- "RCP 8.5"

heatAccum_RCP85 <- heatAccum %>% filter(emission == "RCP 8.5") %>% data.table()
heatAccum_RCP45 <- heatAccum %>% filter(emission == "RCP 4.5") %>% data.table()

daily_CP_RCP85 <- daily_CP %>% filter(emission == "RCP 8.5") %>% data.table()
daily_CP_RCP45 <- daily_CP %>% filter(emission == "RCP 4.5") %>% data.table()

heatAccum_RCP85 <- rbind(heatAccum_RCP85, heatAccum_obs85)
heatAccum_RCP45 <- rbind(heatAccum_RCP45, heatAccum_obs45)

daily_CP_RCP85 <- rbind(daily_CP_RCP85, daily_CP_obs85)
daily_CP_RCP45 <- rbind(daily_CP_RCP45, daily_CP_obs45)

rm(daily_CP, heatAccum)

####################################################################################################
##

app_tp <- "Cripps Pink"

#
# Take care of RCP 8.5. and next do the RCP 4.5
startDoY_letters = c("Sep1", "Sep15", "Oct1", "Oct15", "Nov1", "Nov15", "Dec1", "Dec15", "Jan1")
startDoY = c(1, 15, 31, 46, 62, 77, 92, 107, 123)

print (length(startDoY))
print (length(startDoY_letters))

for (ii in c(1:length(startDoY))){
  curr_startDoY = startDoY[ii]
  curr_startDoY_letter = startDoY_letters[ii]

  print (paste0("curr_startDoY: ", curr_startDoY))
  print (paste0("curr_startDoY_letter: ", curr_startDoY_letter))
  
  curr_heatAccum <- copy(data.table(heatAccum_RCP85))
  curr_daily_CP  <- copy(data.table(daily_CP_RCP85))

  ############################################################
  ## 
  ##   Compute cumulative vertDDs here
  ##
  curr_heatAccum <- curr_heatAccum[chill_dayofyear <= curr_startDoY, vertdd:=0]
  curr_heatAccum <- curr_heatAccum[, vert_Cum_dd := cumsum(vertdd), by=list(location, chill_season, model, emission)]
  curr_heatAccum$vert_Cum_dd_F = curr_heatAccum$vert_Cum_dd * 1.8

  ############################################################
  ##
  ##  pick up needed column so that melts work correctly.
  ##
  ##
  needed_cols <- c("location", "emission", "model", "time_period", "chill_dayofyear")
  
  curr_daily_CP <- subset(curr_daily_CP, select = c(needed_cols, "cume_portions"))
  curr_heatAccum <- subset(curr_heatAccum, select = c(needed_cols, "vert_Cum_dd"))
  ############################################################

  curr_dailyCP_melt <- melt(data.table(curr_daily_CP), id = needed_cols)
  curr_heatAccum <-  melt(data.table(curr_heatAccum) , id = needed_cols)

  merged_dt <- rbind(curr_dailyCP_melt, curr_heatAccum)

  source(bloom_core_source)
  source(bloom_plot_core_source)
  source(chill_core_source)
  x_Max=185
  merged_plt <- double_cloud_2_rows_Accum_VertDD_CP_NoSlopes_200TrySpringerSub2_XCut(d1 = merged_dt,
                                                                                     full_CP_intcpt = 73.3,
                                                                                     heat_intcpt = 25, 
                                                                                     xMax=x_Max)

  plot_dir <- paste0(plot_base_dir ) # "CPAccum_heatAccum/"

  if (dir.exists(plot_dir) == F) {
    dir.create(path = plot_dir, recursive = T)
    print (plot_dir)
  }
  
  Fname <- paste0("00_Acum_CPHeat_", gsub(" ", "_", app_tp), "_RCP85_heatBegins", curr_startDoY_letter, "_ModHist.pdf")
  Fname <- paste0("00_Acum_CPHeat_", "RCP85_heatBegins", curr_startDoY_letter, "_ModHist_Xcut", x_Max, ".pdf")
  ggsave(plot = merged_plt,
         filename = Fname,
         width = 11, height=10, units = "in", 
         dpi = 400, device = "pdf",
         path = plot_dir)
}

#
# Take care of RCP 4.5, and previous one RCP 8.5.
#
for (ii in c(1:length(startDoY))){
  curr_startDoY = startDoY[ii]
  curr_startDoY_letter = startDoY_letters[ii]

  print (paste0("curr_startDoY: ", curr_startDoY))
  print (paste0("curr_startDoY_letter: ", curr_startDoY_letter))
  
  curr_heatAccum <- copy(data.table(heatAccum_RCP45))
  curr_daily_CP  <- copy(data.table(daily_CP_RCP45))

  ############################################################
  ## 
  ##   Compute cumulative vertDDs here
  ##
  curr_heatAccum <- curr_heatAccum[chill_dayofyear <= curr_startDoY, vertdd:=0]
  curr_heatAccum <- curr_heatAccum[, vert_Cum_dd := cumsum(vertdd), by=list(location, chill_season, model, emission)]
  curr_heatAccum$vert_Cum_dd_F = curr_heatAccum$vert_Cum_dd * 1.8

  ############################################################
  ##
  ##  pick up needed column so that melts work correctly.
  ##
  ##
  needed_cols <- c("location", "emission", "model", "time_period", "chill_dayofyear")
  
  curr_daily_CP <- subset(curr_daily_CP, select = c(needed_cols, "cume_portions"))
  curr_heatAccum <- subset(curr_heatAccum, select = c(needed_cols, "vert_Cum_dd"))
  ############################################################

  curr_dailyCP_melt <- melt(data.table(curr_daily_CP), id = needed_cols)
  curr_heatAccum <-  melt(data.table(curr_heatAccum) , id = needed_cols)

  merged_dt <- rbind(curr_dailyCP_melt, curr_heatAccum)

  source(bloom_core_source)
  source(bloom_plot_core_source)
  source(chill_core_source)
  merged_plt <- double_cloud_2_rows_Accum_VertDD_CP_NoSlopes_200TrySpringerSub2_XCut(d1 = merged_dt,
                                                                                     full_CP_intcpt=73.3,
                                                                                     heat_intcpt=25,
                                                                                     xMax=x_Max)

  plot_dir <- paste0(plot_base_dir) # "CPAccum_heatAccum/"

  if (dir.exists(plot_dir) == F) {
    dir.create(path = plot_dir, recursive = T)
    print (plot_dir)
  }
  
  Fname <- paste0("00_Acum_CPHeat_", gsub(" ", "_", app_tp), "_RCP45_heatBegins", curr_startDoY_letter, "_ModHist.pdf")
  Fname <- paste0("00_Acum_CPHeat_", "RCP45_heatBegins", curr_startDoY_letter, "_ModHist_Xcut", x_Max, ".pdf")
  ggsave(plot = merged_plt,
         filename = Fname,
         width = 11, height=10, units = "in", 
         dpi = 400, device = "pdf",
         path = plot_dir)
}


print(Sys.time() - start_time)


