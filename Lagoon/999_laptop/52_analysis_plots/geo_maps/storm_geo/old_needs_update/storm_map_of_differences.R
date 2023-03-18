rm(list=ls())
library(lubridate)
library(ggpubr)
library(purrr)
library(tidyverse)
library(data.table)
library(dplyr)
library(ggplot2)

source_path_1 = "/Users/hn/Documents/GitHub/Ag/Lagoon/core_lagoon.R"
source_path_2 = "/Users/hn/Documents/GitHub/Ag/Lagoon/core_plot_lagoon.R"
source(source_path_1)
source(source_path_2)

options(digit=9)
options(digits=9)
########################################################################
########################################################################
in_dir <- "/Users/hn/Desktop/Desktop/Ag/check_point/lagoon/storm/"
diff_plot_dir <- paste0(in_dir, "new_2_storm/geo_map/diffs/")
if (dir.exists(diff_plot_dir) == F) {dir.create(path = diff_plot_dir, recursive = T)}
print (diff_plot_dir)
########################################################################
all_storms <- readRDS(paste0(in_dir, "all_storms.rds"))
head(all_storms, 2)

all_storms <- all_storms %>%
              filter(return_period != "2006-2025")%>%
              data.table()

all_storms <- within(all_storms, 
                    remove(five_years, ten_years, 
                           fifteen_years, twenty_years))

all_storms <- convert_5_numeric_clusts_to_alphabet(data_tb=all_storms)
########################################################################
clusters <- sort(unique(all_storms$cluster))

clust <- clusters[4]
emissions <- sort(unique(all_storms$emission))
########################################################################
####                   
####     WE DO CLUSTERS SEPARATELY SO COLORs are VISIBLE!
####
########################################################################
for (clust in clusters){
  curr_dt <- all_storms %>% filter(cluster == clust)
  biased_dt <- storm_diff_obs_or_modeled(dt_dt=curr_dt, 
                                         diff_from="1979-2016")

  unbiased_dt <- storm_diff_obs_or_modeled(dt_dt=curr_dt, 
                                           diff_from="1950-2005")
  biased_dt <- biased_dt %>%
               group_by(location, emission, return_period, cluster) %>% 
               summarise(perc_diff_meds = median(perc_diff)) %>% 
               data.table()

  unbiased_dt <- unbiased_dt %>%
                 group_by(location, emission, return_period, cluster) %>% 
                 summarise(perc_diff_meds = median(perc_diff)) %>% 
                 data.table()

  biased_min <- min(biased_dt$perc_diff_meds)
  biased_max <- max(biased_dt$perc_diff_meds)

  unbiased_min <- min(unbiased_dt$perc_diff_meds)
  unbiased_max <- max(unbiased_dt$perc_diff_meds)

  bias_clr_lim <- c(floor(biased_min), ceiling(biased_max))
  unbias_clr_lim <- c(floor(unbiased_min), ceiling(unbiased_max))

  time_ps <- sort(unique(biased_dt$return_period))
  tp <- time_ps[1]
  em <- emissions[1]
  for (em in emissions){
    for (tp in time_ps){
      plt_dt_bias <- biased_dt %>% 
                     filter(emission == em & return_period==tp) %>% 
                     data.table()

      plt_dt_unbias <- unbiased_dt %>% 
                       filter(emission == em & return_period==tp) %>% 
                       data.table()

      assign(x=paste0("bias_", 
                      gsub("[.]", "", gsub("\ ", "", tolower(em))),
                      "_", gsub("-", "_", tp)),
             value = geo_map_perc_diff(dt_dt = plt_dt_bias,
                                       col_col = "perc_diff_meds", 
                                       color_limit = bias_clr_lim) + 
                     ggtitle(label=clust))
                                       
      assign(x = paste0("unbias_", 
                        gsub("[.]", "", gsub("\ ", "", tolower(em))),
                        "_", gsub("-", "_", tp)),
             value = geo_map_perc_diff(dt_dt = plt_dt_unbias,
                                       col_col = "perc_diff_meds", 
                                       color_limit = unbias_clr_lim) + 
                     ggtitle(label=clust))
    }
  }
  assign(x = paste0("bias_45_", gsub("\ ", "_", clust)),
         value = ggarrange(plotlist = list(bias_rcp45_2026_2050,
                                           bias_rcp45_2051_2075,
                                           bias_rcp45_2076_2099),
                           ncol=3, nrow=1, common.legend=FALSE))

  assign(x = paste0("unbias_45_", gsub("\ ", "_", clust)),
         value = ggarrange(plotlist = list(unbias_rcp45_2026_2050,
                                           unbias_rcp45_2051_2075,
                                           unbias_rcp45_2076_2099),
                           ncol=3, nrow=1, common.legend=FALSE))

  assign(x = paste0("bias_85_", gsub("\ ", "_", clust)),
         value = ggarrange(plotlist = list(bias_rcp85_2026_2050,
                                           bias_rcp85_2051_2075,
                                           bias_rcp85_2076_2099),
                           ncol=3, nrow=1, common.legend=FALSE))

  assign(x = paste0("unbias_85_", gsub("\ ", "_", clust)),
         value = ggarrange(plotlist = list(unbias_rcp85_2026_2050,
                                           unbias_rcp85_2051_2075,
                                           unbias_rcp85_2076_2099),
                           ncol=3, nrow=1, common.legend=FALSE))
}

bias_85 <- ggarrange(plotlist = list(bias_85_Western_coastal,
                                     bias_85_Cascade_foothills,
                                     bias_85_Northwest_Cascades,
                                     bias_85_Northcentral_Cascades,
                                     bias_85_Northeast_Cascades),
                     ncol=1, nrow=5, common.legend=FALSE)

bias_45 <- ggarrange(plotlist = list(bias_45_Western_coastal,
                                     bias_45_Cascade_foothills,
                                     bias_45_Northwest_Cascades,
                                     bias_45_Northcentral_Cascades,
                                     bias_45_Northeast_Cascades),
                     ncol=1, nrow=5, common.legend=FALSE)

unbias_85 <- ggarrange(plotlist = list(unbias_85_Western_coastal,
                                       unbias_85_Cascade_foothills,
                                       unbias_85_Northwest_Cascades,
                                       unbias_85_Northcentral_Cascades,
                                       unbias_85_Northeast_Cascades),
                     ncol=1, nrow=5, common.legend=FALSE)

unbias_45 <- ggarrange(plotlist = list(unbias_45_Western_coastal,
                                       unbias_45_Cascade_foothills,
                                       unbias_45_Northwest_Cascades,
                                       unbias_45_Northcentral_Cascades,
                                       unbias_45_Northeast_Cascades),
                     ncol=1, nrow=5, common.legend=FALSE)

bias_45 <- annotate_figure(bias_45,
           top = text_grob(" ", color = "red", face = "bold", size = 14),
           fig.lab = "RCP 4.5 - 25-year/24-hour", fig.lab.face = "bold", 
           fig.lab.size = 14)

bias_85 <- annotate_figure(bias_85,
           top = text_grob(" ", color = "red", face = "bold", size = 14),
           fig.lab = "RCP 8.5 - 25-year/24-hour", fig.lab.face = "bold", 
           fig.lab.size = 14)

unbias_45 <- annotate_figure(unbias_45,
                             top = text_grob(" ", 
                                             color = "red", face = "bold", 
                                             size=14),
                             fig.lab="RCP 4.5 - 25-year/24-hour", 
                             fig.lab.face="bold", fig.lab.size=14)

unbias_85 <- annotate_figure(unbias_85,
                             top = text_grob(" ", 
                                             color = "red", face = "bold", 
                                             size = 14),
                             fig.lab="RCP 8.5 - 25-year/24-hour", 
                             fig.lab.face = "bold", fig.lab.size=14)

ggsave(filename = paste0("bias_45_sep_clust.png"), 
       plot=bias_45, 
       width=7, height=10, units="in", 
       dpi=600, device="png", path=diff_plot_dir)

ggsave(filename = paste0("bias_85_sep_clust.png"), 
       plot=bias_85, 
       width=7, height=10, units="in", 
       dpi=600, device="png", path=diff_plot_dir)

ggsave(filename = paste0("unbias_45_sep_clust.png"), 
       plot=unbias_45, 
       width=7, height=10, units="in", 
       dpi=600, device="png", path=diff_plot_dir)

ggsave(filename = paste0("unbias_85_sep_clust.png"), 
       plot=unbias_85, 
       width=7, height=10, units="in", 
       dpi=600, device="png", path=diff_plot_dir)


