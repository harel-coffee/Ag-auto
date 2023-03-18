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
source_dir <- "/Users/hn/Documents/00_GitHub/Ag/Bloom/"
in_dir <- "/Users/hn/Documents/01_research_data/bloom/"
param_dir <- paste0(source_dir, "parameters/")
#############################################################
#############################################################

source_1 <- paste0(source_dir, "bloom_core.R")
source_2 <- paste0(source_dir, "bloom_plot_core.R")
source(source_1)
source(source_2)
#############################################################
###
###               Read data off the disk
###
#############################################################
limited_locations <- read.csv(file = paste0(param_dir, "limited_locations.csv"), 
                        header=TRUE, as.is=TRUE)

limited_locations$location <- paste0(limited_locations$lat, "_", limited_locations$long)
limited_locations <- within(limited_locations, remove(lat, long))

limited_locations <- limited_locations %>% 
                     filter(city %in% c("Eugene", "Walla Walla")) # , "Walla Walla"

#############################################################
bloom <- readRDS(paste0(in_dir, "fullbloom_50percent_day.rds"))
thresh <- readRDS(paste0(in_dir, "sept_summary_comp.rds"))

#############################################################
#
# pick up observed and 2026-2099 time period
#
#############################################################

bloom <- bloom %>% filter(chill_season >= "chill_2026-2027") %>% data.table()
thresh <- thresh %>% filter(chill_season >= "chill_2026-2027") %>% data.table()

bloom$location <- paste0(bloom$lat, "_", bloom$long)
bloom <- bloom %>% filter(location %in% limited_locations$location) %>% data.table()
thresh <- thresh %>% filter(location %in% limited_locations$location) %>% data.table()

bloom <- bloom %>% filter(fruit_type == "gala") %>% data.table()
bloom <- dplyr::left_join(x = bloom, y = limited_locations, by = "location")
thresh <- dplyr::left_join(x = thresh, y = limited_locations, by = "location")

#############################################################
#
#              clean up each data table
#
#############################################################
bloom <- within(bloom, remove(month, day, dayofyear, bloom_perc, lat, long, 
                              location, year, time_period, fruit_type))

thresh <- within(thresh, remove(lat, long, time_period, thresh_20, 
                                thresh_25, thresh_30, thresh_35, thresh_40,
                                thresh_50, thresh_55, thresh_60, thresh_65, thresh_70, location))

bloom <- data.table(bloom)
thresh <- data.table(thresh)

v <- c("chill_season", "city", "emission", "model", "chill_doy")
setcolorder(bloom, v)

v <- c("chill_season", "city", "emission", "model", "thresh_45", "thresh_75")
setcolorder(thresh, v)

setnames(bloom, old=c("chill_doy"), new=c("chill_doy_of_bloom"))

thresh$model <- gsub("-", "_", thresh$model)


fruit_type <- "apple"
if (fruit_type == "cherry"){
  bloom$chill_doy_of_bloom <- bloom$chill_doy_of_bloom - 14
}

thresh_bloom_dt <- dplyr::left_join(x = thresh, y = bloom)

bloom_thresh_cross_TS <- function(thresh_bloom){
  thresh_bloom$bloom_minus_75CP <- thresh_bloom$chill_doy_of_bloom - thresh_bloom$thresh_75
  thresh_bloom$bloom_minus_45CP <- thresh_bloom$chill_doy_of_bloom - thresh_bloom$thresh_45

  thresh_bloom <- data.table(thresh_bloom)

  thresh_bloom_75CP <- within(thresh_bloom, remove("model", "thresh_45", "thresh_75", 
                                                   "chill_doy_of_bloom", "bloom_minus_45CP"))


  thresh_bloom_45CP <- within(thresh_bloom, remove("model", "thresh_45", "thresh_75", 
                                                   "chill_doy_of_bloom", "bloom_minus_75CP"))

  ######
  ######  pick rows that represent intersection. chill_doy_of_bloom - CP_DoY < 0
  ######  (i.e. bloom is earlier than CP is accumulated)
  ######
  thresh_bloom_75CP <- thresh_bloom_75CP %>% filter(bloom_minus_75CP <= 0 ) %>% data.table()
  thresh_bloom_45CP <- thresh_bloom_45CP %>% filter(bloom_minus_45CP <= 0 ) %>% data.table()

  thresh_bloom_75CP <- thresh_bloom_75CP %>%
                       group_by(chill_season, city, emission) %>%
                       summarise(count=n()) %>%
                       data.table()

  thresh_bloom_45CP <- thresh_bloom_45CP %>%
                       group_by(chill_season, city, emission) %>%
                       summarise(count=n()) %>%
                       data.table()

  #
  # In the data tables thresh_bloom_75CP and thresh_bloom_45CP some
  # of the chill seasons would be missing, if the difference were not negative
  # so, we need to recreate them and set them to zero! 
  #
  out_df_75CP <- CJ(unique(bloom$city), unique(bloom$chill_season), unique(bloom$emission))
  colnames(out_df_75CP) <- c("city", "chill_season", "emission")

  out_df_45CP <- out_df_75CP

  out_df_75CP <- dplyr::left_join(x = out_df_75CP, y = thresh_bloom_75CP)
  out_df_75CP[is.na(out_df_75CP)] <- 0

  out_df_45CP <- dplyr::left_join(x = out_df_45CP, y = thresh_bloom_45CP)
  out_df_45CP[is.na(out_df_45CP)] <- 0

  return(list(out_df_75CP, out_df_45CP))
}

cross_over_TS_plotting <- function(d1, fil="cross overs"){
  d1$chill_season <- gsub("chill_", "", d1$chill_season)
  d1$chill_season <- substr(d1$chill_season, 1, 4)
  d1$chill_season <- as.numeric(d1$chill_season)
  
  
  xbreaks <- seq(2025, 2085, 15)
  ybreaks <- seq(0, 18, 4)

  ict <- c("Omak", "Yakima", "Walla Walla", "Eugene")
  d1$city <- factor(d1$city, levels = ict, order=TRUE)

  d1$emission <- factor(d1$emission, 
                        levels = c("RCP 8.5", "RCP 4.5"), order=TRUE)

  
  ggplot(d1, aes(x=chill_season, y=count, fill=fil)) +
  labs(x = "chill year", y = "number of crossovers") + 
  facet_grid( ~ emission ~ city) + # scales = "free"
  geom_line() + 
  scale_x_continuous(breaks = xbreaks) +
  scale_y_continuous(breaks = ybreaks) +
  theme(panel.grid.major = element_line(size=0.2),
        panel.spacing=unit(.5, "cm"),
        legend.text=element_text(size=18, face="bold"),
        legend.title = element_blank(),
        legend.position = "bottom",
        strip.text = element_text(face="bold", size=16, color="black"),
        axis.text = element_text(size=16, color="black"), # face="bold",
        # axis.text.x = element_text(angle=20, hjust = 1),
        axis.ticks = element_line(color = "black", size = .2),
        axis.title.x = element_text(size=18,  face="bold", 
                                    margin=margin(t=10, r=0, b=0, l=0)),
        axis.title.y = element_text(size=18, face="bold",
                                    margin=margin(t=0, r=10, b=0, l=0)),
        plot.title = element_text(lineheight=.8, face="bold", size=20)
        ) 
}


cross_time_series <- bloom_thresh_cross_TS(thresh_bloom_dt)

df_75CP <- cross_time_series[[1]]
df_45CP <- cross_time_series[[2]]

crossOvers_75CP_plot <- cross_over_TS_plotting(d1 = df_75CP, fil="cross overs")
crossOvers_45CP_plot <- cross_over_TS_plotting(d1 = df_45CP, fil="cross overs")

plot_dir <- "/Users/hn/Documents/00_GitHub/Ag_papers/Chill_Paper/MajorRevision/figures_4_revision/crossovers_3rd_params/"
if (dir.exists(plot_dir) == F) {dir.create(path = plot_dir, recursive = T)}

if (length(unique(df_75CP$city)) == 2){
  W = 10
  H = 5
  } else if (length(unique(df_75CP$city)) == 1){
  W = 5
  H = 5
}

ggsave(plot=crossOvers_75CP_plot,
       filename = paste0(fruit_type, "_crossovers_75CP.png"), 
       width=W, height=H, units = "in", 
       dpi=600, device = "png",
       path=plot_dir)

ggsave(plot=crossOvers_45CP_plot,
       filename = paste0(fruit_type, "_crossovers_45CP.png"), 
       width = W, height=H, units = "in", 
       dpi=600, device = "png",
       path = plot_dir)

