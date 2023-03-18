rm(list=ls())
library(lubridate)
library(ggpubr)
library(purrr)
library(tidyverse)
library(data.table)
library(dplyr)
library(ggplot2)
options(digit=9)
options(digits=9)

source_path_1 = "/Users/hn/Documents/GitHub/Ag/Lagoon/core_lagoon.R"
source_path_2 = "/Users/hn/Documents/GitHub/Ag/Lagoon/core_plot_lagoon.R"
source(source_path_1)
source(source_path_2)

#########################
#
#
fi <- "/Users/hn/Desktop/Desktop/Ag/"
sec <- "check_point/lagoon/storm/"
in_dir <- paste0(fi, sec)

AVs_plot_dir <- paste0(in_dir, "new_2_storm/geo_map/AVs/")
if (dir.exists(AVs_plot_dir) == F) {dir.create(path = AVs_plot_dir, recursive = T)}
print (AVs_plot_dir)
###################################################################################
#
param_dir <- "/Users/hn/Documents/GitHub/Ag/Lagoon/parameters/"
fip_clust <- read.csv(paste0(param_dir, "precip_elev_5_clusters.csv"), 
                      header=T, as.is=T)
fip_clust <- convert_5_numeric_clusts_to_alphabet(data_tb=fip_clust)
###################################################################################
#
# Read file
#
storm_file <- "all_storms.rds"
storm <- data.table(readRDS(paste0(in_dir, storm_file)))
storm <- get_ridof_canada(storm)
storm <- convert_5_numeric_clusts_to_alphabet(storm)
#
# Filter modeled hist out
#
# storm <- storm %>%
#          filter(return_period != "2006-2025" & 
#                 return_period != "1950-2005") %>%
#          data.table()
storm <- storm %>%
         filter(return_period != "2006-2025" & 
                return_period != "1979-2016") %>%
         data.table()
######
######
######
# return_levels <- c("1979-2016", "2026-2050", "2051-2075", "2076-2099")
return_levels <- sort(unique(storm$return_period))
future_rn_pr <- c("2026-2050", "2051-2075", "2076-2099")

storm$return_period <- factor(storm$return_period, 
                              levels = return_levels)
#
# pick up the columns that matter:
#
nd_cols <- c("location", "model", "emission", "cluster",
             "return_period", "twenty_five_years")
storm <- subset(storm, select=nd_cols)
storm_mod_hist <- storm %>%
                  filter(return_period == "1950-2005") %>%
                  select(-c("emission")) %>%
                  unique()%>% data.table()
storm_F <- storm %>%
           filter(return_period != "1950-2005") %>%
           data.table()
rm(storm)

####################################
#
#        Medians 
#
####################################
#
# find medians among models for each location
#
storm_F_medians <- storm_F %>%
                   group_by(location, emission, return_period, cluster) %>% 
                   summarise(twenty_five_years = median(twenty_five_years)) %>% 
                   data.table()

storm_mod_hist <- storm_mod_hist %>%
                  group_by(location, return_period, cluster) %>% 
                  summarise(twenty_five_years = median(twenty_five_years)) %>% 
                  data.table()

minnn_h <- min(storm_mod_hist$twenty_five_years)
maxxx_h <- max(storm_mod_hist$twenty_five_years)

minnn_f <- min(storm_F_medians$twenty_five_years)
maxxx_f <- max(storm_F_medians$twenty_five_years)

minnn_all <- min(minnn_h, minnn_f)
maxxx_all <- max(maxxx_h, maxxx_f)

model_hist_plt <- obs_hist_map_storm(dt=storm_mod_hist, 
                                     minn=minnn_all, maxx=maxxx_all, 
                                     fips_clust=fip_clust,
                                     tgt_col="twenty_five_years") + 
                  ggtitle("Modeled historical\n25-year/24-hour design storm")

emissions <- c("RCP 4.5", "RCP 8.5")
em <- emissions[1]; rp <- future_rn_pr[1]

for (em in emissions){
  for (rp in future_rn_pr){
    curr_dt <- storm_F_medians %>%
               filter(emission == em & return_period==rp) %>%
               data.table()
    # minnn <- min(curr_dt$twenty_five_years)
    # maxxx <- max(curr_dt$twenty_five_years)

    title <- paste0("25-year/24-hour design storm",
                    ", ", "actual values", 
                    "\n", em, ", ", rp)

    assign(x = paste0(gsub("[.]", "", gsub("\ ", "", tolower(em))), "_",
                      gsub(pattern = "-", replacement = "_", x = rp)),
           value ={one_time_medians_storm_geoMap(curr_dt, 
                                                 minn=minnn_all, maxx=maxxx_all, 
                                                 ttl=title, subttl=subttl, 
                                                 differ=FALSE)})
  }
}

median_45 <- ggarrange(plotlist = list(model_hist_plt, 
                                       rcp45_2026_2050,
                                       rcp45_2051_2075, 
                                       rcp45_2076_2099),
                       ncol=4, nrow=1, common.legend=FALSE)

median_85 <- ggarrange(plotlist = list(model_hist_plt,
                                       rcp85_2026_2050,
                                       rcp85_2051_2075,
                                       rcp85_2076_2099),
                      ncol=4, nrow=1, common.legend=FALSE)

ggsave(filename = paste0("AV_85_all_clust_14_inch_wide.png"), 
       plot=median_85, 
       width=14, height=4, units="in", 
       dpi=600, device="png", path=AVs_plot_dir)

ggsave(filename = paste0("AV_45_all_clust_14_inch_wide.png"), 
       plot=median_45, 
       width=14, height=4, units="in", 
       dpi=600, device="png", path=AVs_plot_dir)


ggsave(filename = paste0("AV_85_all_clust_16_inch_wide.png"), 
       plot=median_85, 
       width=16, height=4, units="in", 
       dpi=600, device="png", path=AVs_plot_dir)

ggsave(filename = paste0("AV_45_all_clust_16_inch_wide.png"), 
       plot=median_45, 
       width=16, height=4, units="in", 
       dpi=600, device="png", path=AVs_plot_dir)

ggsave(filename = paste0("AV_85_all_clust_18_inch_wide.png"), 
       plot=median_85, 
       width=18, height=4, units="in", 
       dpi=600, device="png", path=AVs_plot_dir)

ggsave(filename = paste0("AV_45_all_clust_18_inch_wide.png"), 
       plot=median_45, 
       width=18, height=4, units="in", 
       dpi=600, device="png", path=AVs_plot_dir)
