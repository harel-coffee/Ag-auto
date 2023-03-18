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

###################################################################################
#
# Read file
#
storm_file <- "all_storms.rds"
storm <- data.table(readRDS(paste0(in_dir, storm_file)))
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
                              levels=return_levels)
#
# pick up the columns that matter:
#
nd_cols <- c("location", "model", "emission", 
             "return_period", "twenty_five_years")
storm <- subset(storm, select=nd_cols)

# storm_obs <- storm %>%
#              filter(return_period == "1979-2016") %>%
#              select(-c("emission")) %>%
#              unique()%>%
#              data.table()
storm_mod_hist <- storm %>%
                  filter(return_period == "1950-2005") %>%
                  select(-c("emission")) %>%
                  unique()%>% data.table()

storm_F <- storm %>%
           filter(return_period != "1950-2005") %>%
           data.table()

storm_45 <- storm_F %>%
            filter(emission == "RCP 4.5")%>%
            data.table()

storm_85 <- storm_F %>%
            filter(emission == "RCP 8.5") %>%
            data.table()
rm(storm)

##
### Plot RCP 4.5 all models
##
minn <- min(storm_45$twenty_five_years)
maxx <- max(storm_45$twenty_five_years)
for (ft_pr in future_rn_pr){
  curr_dt <- storm_45 %>%
             filter(return_period %in% c("1979-2016", ft_pr))%>%
             data.table()
  title <- paste0("RCP 4.5 (", ft_pr, ")")
  
  assign(x = paste0("map_45_", 
                    gsub(pattern = "-", 
                         replacement = "_", 
                         x = ft_pr)),
         value ={all_mods_map_storm(curr_dt, 
                                    minn=minn, maxx=maxx, 
                                    title)})
}
RCP45_figs <- ggarrange(plotlist = list(map_45_2026_2050,
                                        map_45_2051_2075,
                                        map_45_2076_2099),
                        ncol=1, nrow=3)

ggsave(filename = paste0("rcp45_all_map.png"), 
       plot=RCP45_figs, 
       width=8, height=15, units="in", 
       dpi=600, device="png", path=AVs_plot_dir)

ggsave(filename = paste0("map_45_2026_2050.png"), 
       plot = map_45_2026_2050, 
       width=10, height=6, units = "in", 
       dpi=600, device = "png", path = AVs_plot_dir)

ggsave(filename = paste0("map_45_2051_2075.png"), 
       plot = map_45_2051_2075, 
       width=10, height=6, units = "in", 
       dpi=600, device = "png", path = AVs_plot_dir)

ggsave(filename = paste0("map_45_2076_2099.png"), 
       plot = map_45_2076_2099, 
       width=10, height=6, units = "in", 
       dpi=600, device = "png", path = AVs_plot_dir)

rm(min, max, RCP45_figs, map_45_2026_2050,
   map_45_2051_2075, map_45_2076_2099)
#
# Plot RCP 8.5 all models
#
min <- min(storm_85$twenty_five_years)
max <- max(storm_85$twenty_five_years)
for (ft_pr in future_rn_pr){
  curr_dt <- storm_85 %>%
             filter(return_period %in% c("1979-2016", ft_pr))%>%
             data.table()
  title <- paste0("RCP 8.5 (", ft_pr, ")")
  
  assign(x = paste0("map_85_", 
                    gsub(pattern = "-", 
                         replacement = "_", 
                         x = ft_pr)),
         value ={all_mods_map_storm(curr_dt, minn=min, maxx=max, title)})
}
RCP85_figs <- ggarrange(plotlist = list(map_85_2026_2050,
                                        map_85_2051_2075,
                                        map_85_2076_2099),
                        ncol=1, nrow=3)
ggsave(filename = paste0("rcp85_all_map.png"), 
       plot=RCP85_figs, 
       width=8, height=15, units = "in", 
       dpi=600, device = "png", path = AVs_plot_dir)

ggsave(filename = paste0("map_85_2026_2050.png"), 
       plot=map_85_2026_2050, 
       width=10, height=6, units = "in", 
       dpi=600, device="png", path = AVs_plot_dir)

ggsave(filename = paste0("map_85_2051_2075.png"), 
       plot = map_85_2051_2075, 
       width=10, height=6, units = "in", 
       dpi=600, device = "png", path = AVs_plot_dir)

ggsave(filename = paste0("map_85_2076_2099.png"), 
       plot = map_85_2076_2099, 
       width=10, height=6, units = "in", 
       dpi=600, device = "png", path = AVs_plot_dir)

rm(min, max, RCP85_figs,
   map_85_2026_2050, map_85_2051_2075, map_85_2076_2099)

####################################
#
#        Medians 
#
####################################
#
# find medians among models for each location
#
storm_F_medians <- storm_F %>%
                   group_by(location, emission, return_period) %>% 
                   summarise(twenty_five_years = median(twenty_five_years)) %>% 
                   data.table()

storm_mod_hist <- storm_mod_hist %>%
                  group_by(location, return_period) %>% 
                  summarise(twenty_five_years = median(twenty_five_years)) %>% 
                  data.table()
#
# Plot all medians
#
########################################
#
#       Map of observed data
#
########################################
minnn <- min(storm_mod_hist$twenty_five_years)
maxxx <- max(storm_mod_hist$twenty_five_years)

minnn <- min(minnn, min(storm_F_medians$twenty_five_years))
maxxx <- max(maxxx, max(storm_F_medians$twenty_five_years))

storm_mod_hist_plt <- obs_hist_map_storm(dt=storm_mod_hist, 
                                         minn=minnn, maxx=maxxx, 
                                         fips_clust=fip_clust, 
                                         tgt_col="twenty_five_years") + 
                      ggtitle("Modeled historical")

emissions <- c("RCP 4.5", "RCP 8.5")
# subttl <- "medians taken over models"
em <- emissions[1]; rp <- future_rn_pr[1]

for (em in emissions){
  for (rp in future_rn_pr){
    curr_dt <- storm_F_medians %>%
               filter(emission == em & return_period==rp) %>%
               data.table()
    title <- paste0(em, " (", rp, ")")
    assign(x = paste0(gsub(pattern = " ", 
                           replacement = "_", 
                           x = em), "_",
                      gsub(pattern = "-", replacement = "_", x = rp)),
           value ={one_time_medians_storm_geoMap(curr_dt, 
                                                 minn=minnn, maxx=maxxx, 
                                                 ttl=title, subttl=subttl, 
                                                 differ=FALSE)})

  }
}

median_figs <- ggarrange(plotlist = list(RCP_8.5_2076_2099,
                                         RCP_8.5_2051_2075,
                                         RCP_8.5_2026_2050,
                                         RCP_4.5_2076_2099,
                                         RCP_4.5_2051_2075,
                                         RCP_4.5_2026_2050,
                                         NULL, storm_mod_hist_plt, NULL),
                         ncol=3, nrow=3, common.legend = TRUE,
                         legend="bottom")

ggsave(filename = "median_figs.png", 
       plot = median_figs, 
       width=6, height=4, units = "in", 
       dpi=300, device="png", path = AVs_plot_dir)

median_figs_85 <- ggarrange(plotlist = list(storm_mod_hist_plt, 
                                            RCP_8.5_2026_2050,
                                            RCP_8.5_2051_2075,
                                            RCP_8.5_2076_2099),
                            ncol=4, nrow=1, common.legend=TRUE,
                            legend="bottom")

ggsave(filename = "median_figs_85.png", plot=median_figs_85, 
       width=7, height=2, units = "in", 
       dpi=300, device = "png", path = AVs_plot_dir)

median_figs_45 <- ggarrange(plotlist = list(storm_mod_hist_plt, 
                                            RCP_4.5_2026_2050,
                                            RCP_4.5_2051_2075,
                                            RCP_4.5_2076_2099),
                            ncol=4, nrow=1, common.legend=TRUE, 
                            legend="bottom")

ggsave(filename = "median_figs_45.png", plot = median_figs_45, 
       width=7, height=2, units="in", 
       dpi=300, device="png", path=AVs_plot_dir)

rm(RCP_8.5_2076_2099, RCP_8.5_2051_2075, RCP_8.5_2026_2050,
   RCP_4.5_2076_2099, RCP_4.5_2051_2075, RCP_4.5_2026_2050,
   median_figs)
###########################################
#
# Differences of median and obs
#
###########################################
# storm_diff <- storm_F_medians
# for (row in 1:nrow(storm_diff)){
#   curr_loc <- storm_F_medians[row, location]
#   curr_hist_val <- storm_obs[location==curr_loc, twenty_five_years]
#   storm_diff[row, "twenty_five_years"] <- storm_diff[row, twenty_five_years] - curr_hist_val
# }
# saveRDS(storm_diff, paste0(in_dir, "storm_medians_diff_25yrs.rds"))



# storm_diff <- readRDS(paste0(in_dir, "storm_medians_diff_25yrs.rds"))

# min <- min(storm_diff$twenty_five_years)
# max <- max(storm_diff$twenty_five_years)

# for (em in emissions){
#   for (rp in future_rn_pr){
#     curr_dt <- storm_diff %>%
#                filter(emission == em & return_period==rp) %>%
#                data.table()
#     title <- paste0(em, " (", rp, ")")
#     assign(x = paste0(gsub(pattern = " ", 
#                            replacement = "_", 
#                            x = em),
#                       "_",
#                       gsub(pattern = "-", 
#                            replacement = "_", 
#                            x = rp)),
#            value ={one_time_medians_storm_geoMap(curr_dt, minn = min, maxx = max, 
#                                                  title, subttl, differ=TRUE)})

#   }
# }
# diff_median_figs <- ggarrange(plotlist = list(RCP_8.5_2076_2099,
#                                               RCP_8.5_2051_2075,
#                                               RCP_8.5_2026_2050,
#                                               RCP_4.5_2076_2099,
#                                               RCP_4.5_2076_2099,
#                                               RCP_4.5_2076_2099),
#                         ncol = 3, nrow = 2,
#                         common.legend = TRUE)

# ggsave(filename = "diff_median_figs.png", 
#        plot = diff_median_figs, 
#        width = 10, height = 7, units = "in", 
#        dpi=300, device = "png",
#        path = plot_dir)
