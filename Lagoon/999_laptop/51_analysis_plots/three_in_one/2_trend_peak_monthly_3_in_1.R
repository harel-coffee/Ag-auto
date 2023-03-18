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
#
# This is a lost cause. sick, set in specific ways, 
# jumping to conclusions.
# not spending time to think.
#
source_path_1 = "/Users/hn/Documents/GitHub/Ag/Lagoon/core_lagoon.R"
source_path_2 = "/Users/hn/Documents/GitHub/Ag/Lagoon/core_plot_lagoon.R"
source(source_path_1)
source(source_path_2)
start_time <- Sys.time()
##############################################################
base <- "/Users/hn/Desktop/Desktop/Ag/check_point/lagoon/"
data_base <- paste0(base, "rain_snow_fractions/")
diff_dir <- paste0(base, "/precip/02_med_diff_med_no_bias/")
in_dir <- data_base
##############################################################
AV_title <- "monthly precip."
AV_y_lab <- "cum. precip. (mm)"
AV_tg_col <- "monthly_cum_precip"

AVs <- readRDS(paste0(in_dir, "monthly_fracs.rds")) %>% data.table()
AVs <- remove_observed(AVs)
AVs <- remove_current_timeP(AVs) # remove 2006-2025
# update clusters labels
AVs <- convert_5_numeric_clusts_to_alphabet(data_tb = AVs)
AVs <- na.omit(AVs)
AVs <- within(AVs, remove(day, rain_portion, 
                          monthly_cum_rain, monthly_cum_snow))
AVs$rain_fraction <- AVs$rain_fraction * 100
AVs_45 <- AVs %>% filter(emission=="RCP 4.5") %>% data.table()
AVs_85 <- AVs %>% filter(emission=="RCP 8.5") %>% data.table()
############################################################
unbias_diff<-readRDS(paste0(diff_dir, 
                     "detail_med_diff_med_month_precip.rds")) %>% 
               data.table()
unbias_diff <- na.omit(unbias_diff)
unbias_diff <- remove_observed(unbias_diff)
unbias_diff <- remove_current_timeP(unbias_diff)#remove 2006-2025
# update clusters labels
unbias_diff<-convert_5_numeric_clusts_to_alphabet(unbias_diff)
unbias_diff_45 <- unbias_diff %>% 
                  filter(emission=="RCP 4.5") %>% 
                  data.table()
unbias_diff_85 <- unbias_diff %>% 
                  filter(emission=="RCP 8.5") %>% 
                  data.table() 
rm(unbias_diff)
############################################################
timeP_ty <- 1
cluster_types <- unique(AVs$cluster)
clust_g <- cluster_types[1]

for (clust_g in cluster_types){
  print(clust_g)
  subttl <- paste0(" (", clust_g, ")")
  curr_AVs_45 <- AVs_45 %>% filter(cluster == clust_g) %>% data.table()
  curr_AVs_85 <- AVs_85 %>% filter(cluster == clust_g) %>% data.table()

  curr_diff_85 <- unbias_diff_85 %>% 
                  filter(cluster == clust_g) %>% 
                  data.table()
  curr_diff_45 <- unbias_diff_45 %>% 
                  filter(cluster == clust_g) %>% 
                  data.table()
  #########
  ######### Actual value plots
  #########
  quans_85 <- find_quantiles(curr_AVs_85, 
                             tgt_col= AV_tg_col, 
                            time_type="monthly")
  quans_45 <- find_quantiles(curr_AVs_45, 
                             tgt_col= AV_tg_col, 
                             time_type="monthly")

  AV_box_85 <- box_trend_monthly_cum(dt = curr_AVs_85, p_type="box",
                                     y_lab = AV_y_lab, 
                                     tgt_col = AV_tg_col) +
               ggtitle(paste0(AV_title)) + 
               coord_cartesian(ylim = c(quans_85[1], quans_85[2]))

  AV_box_45 <- box_trend_monthly_cum(dt = curr_AVs_45, p_type="box",
                                     y_lab = AV_y_lab, 
                                     tgt_col = AV_tg_col) + 
               ggtitle(paste0(AV_title)) + 
               coord_cartesian(ylim = c(quans_45[1], quans_45[2]))
  #########
  ######### difference plot
  #########
  a1 <- "percentage differences"
  a2 <- " between future time periods and historical"
  box_title <- paste0(a1, a2)
  quans_85 <- find_quantiles(curr_diff_85, 
                             tgt_col= "perc_diff", 
                             time_type="monthly")
  quans_45 <- find_quantiles(curr_diff_45, 
                             tgt_col= "perc_diff", 
                             time_type="monthly")

  unbias_perc_diff_85 <- box_trend_monthly_cum(dt=curr_diff_85,
                                               p_type="box",
                                               y_lab="differences (%)",
                                               tgt_col="perc_diff") + 
                         ggtitle(box_title) +
                         coord_cartesian(ylim=c(quans_85[1], quans_85[2]))

  unbias_perc_diff_45 <- box_trend_monthly_cum(dt=curr_diff_45,
                                               p_type="box",
                                               y_lab="differences (%)",
                                               tgt_col="perc_diff") + 
                         ggtitle(box_title) + 
                         coord_cartesian(ylim=c(quans_45[1], quans_45[2]))
  #########
  ######### rain fractions
  #########
  box_title <- paste0("fraction of precip. fell as rain")
  quans_85 <- find_quantiles(curr_AVs_85, 
                             tgt_col="rain_fraction", 
                             time_type="monthly")
  quans_45 <- find_quantiles(curr_AVs_45, 
                             tgt_col="rain_fraction",
                             time_type="monthly")
             
  rain_frac_85 <- box_trend_monthly_cum(dt = curr_AVs_85, p_type="box",
                                        y_lab="rain fraction (%)", 
                                        tgt_col="rain_fraction") + 
                  ggtitle(box_title) + 
                  coord_cartesian(ylim = c(max(-2, quans_85[1]), 
                                         min(105, quans_85[2])))
  ####################################################################
  rain_frac_45 <- box_trend_monthly_cum(dt=curr_AVs_45, p_type="box",
                                        y_lab="rain fraction (%)", 
                                        tgt_col="rain_fraction") + 
                  ggtitle(box_title) + 
                  coord_cartesian(ylim = c(max(-2, quans_45[1]), 
                                         min(105, quans_45[2])))
  #################################################################
  ######################################################
  ########
  ########   just frac 
  ########
  DD <- "narrowed_rain_snow_fractions/monthly/just_frac/"
  just_frac_plt_dir <- paste0(data_base, DD)
  if (dir.exists(just_frac_plt_dir) == F){
     dir.create(path = just_frac_plt_dir, recursive = T)}

  ggsave(filename = paste0(clust_g, "_rain_85.png"),
         plot=rain_frac_85, width=5.5, height=2, units = "in", 
         dpi=600, device = "png", path = just_frac_plt_dir)

  ggsave(filename = paste0(clust_g, "_rain_45.png"),
         plot = rain_frac_45, width=5.5, height=2, units = "in", 
         dpi=600, device = "png", path=just_frac_plt_dir)
  ######################################################
  ########
  ########   just AVs
  ########
  DD <- "narrowed_rain_snow_fractions/monthly/just_AVs/"
  just_av_plt_dir <- paste0(data_base, DD)
  if (dir.exists(just_av_plt_dir) == F) {
    dir.create(path = just_av_plt_dir, recursive = T)}
  
  ggsave(filename = paste0(clust_g, "_rain_85.png"),
         plot = AV_box_85, width=5.5, height=2, units = "in", 
         dpi=600, device = "png", path = just_av_plt_dir)

  ggsave(filename = paste0(clust_g, "_rain_45.png"),
         plot = AV_box_45, width=5.5, height=2, units = "in", 
         dpi=600, device = "png", path = just_av_plt_dir)

  ######################################################
  ########
  ########   just diffs
  ########
  DD <- "narrowed_rain_snow_fractions/monthly/just_diff/"
  just_diff_plt_dir <- paste0(data_base, DD)
  if (dir.exists(just_diff_plt_dir) == F) {
    dir.create(path = just_diff_plt_dir, recursive = T)}
  
  ggsave(filename = paste0(clust_g, "_rain_85.png"),
         plot = unbias_perc_diff_85, width=7, height=2, units = "in", 
         dpi=600, device = "png", path = just_diff_plt_dir)

  ggsave(filename = paste0(clust_g, "_rain_45.png"),
         plot = unbias_perc_diff_45, width=7, height=2, units = "in", 
         dpi=600, device = "png", path = just_diff_plt_dir) 
  #################################################################
  DD <- "narrowed_rain_snow_fractions/monthly/3_in_1/"
  plot_dir <- paste0(data_base, DD)
  if (dir.exists(plot_dir) == F) {dir.create(path = plot_dir, recursive = T)}
  print (plot_dir)

  three_85 <- ggarrange(plotlist = list(AV_box_85, 
                                        unbias_perc_diff_85, 
                                        rain_frac_85),
                       ncol=1, nrow=3, common.legend = TRUE, 
                       legend="bottom")
  
  three_45 <- ggarrange(plotlist = list(AV_box_45, 
                                        unbias_perc_diff_45, 
                                        rain_frac_45),
                       ncol=1, nrow=3, common.legend = TRUE, 
                       legend="bottom")
  
  ggsave(filename = paste0(clust_g, "_rain_85.png"),
         plot = three_85, width=7, height=5, units = "in", 
         dpi=600, device = "png", path = plot_dir)
  
  ggsave(filename = paste0(clust_g, "_rain_45.png"),
         plot = three_45, width =7, height=5, units = "in", 
         dpi=600, device = "png", path = plot_dir)
}

print (Sys.time() - start_time)


