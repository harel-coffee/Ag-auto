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

start_time <- Sys.time()
##############################################################
base <- "/Users/hn/Desktop/Desktop/Ag/check_point/lagoon/"
data_base <- paste0(base, "rain_snow_fractions/")
diff_dir <- paste0(base, "/precip/02_med_diff_med_no_bias/")
in_dir <- data_base
plot_base <- paste0(base, "plots/precip/monthly/")
##############################################################
AV_title <- paste0("Total monthly precipitation")

AV_y_lab <- "precipitation (mm)"
AV_tg_col <- "monthly_cum_precip"

AVs <- readRDS(paste0(in_dir, "monthly_fracs.rds")) %>% 
       data.table()
AVs <- remove_observed(AVs)
AVs <- remove_current_timeP(AVs) # remove 2006-2025
# update clusters labels
AVs <- convert_5_numeric_clusts_to_alphabet(AVs)
AVs <- na.omit(AVs)
AVs <- within(AVs, remove(day, rain_portion, 
                          monthly_cum_rain, monthly_cum_snow))

AVs$rain_fraction <- AVs$rain_fraction * 100
AVs_45 <- AVs %>% filter(emission=="RCP 4.5") %>% data.table()
AVs_85 <- AVs %>% filter(emission=="RCP 8.5") %>% data.table()
############################################################
unbias_diff <- readRDS(paste0(diff_dir, 
                              "detail_med_diff_med_month_precip.rds")) %>% 
               data.table()

unbias_diff <- na.omit(unbias_diff)
unbias_diff <- remove_observed(unbias_diff)
unbias_diff <- remove_current_timeP(unbias_diff)
# update clusters labels

unbias_diff <- convert_5_numeric_clusts_to_alphabet(unbias_diff)

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
  curr_AVs_45 <- AVs_45 %>% 
                 filter(cluster == clust_g) %>% 
                 data.table()

  curr_AVs_85 <- AVs_85 %>% 
                 filter(cluster == clust_g) %>% 
                 data.table()

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

  AV_box_85 <- box_trend_monthly_cum(dt = curr_AVs_85, 
                                     p_type="box",
                                     y_lab = AV_y_lab, 
                                     tgt_col = AV_tg_col) +
               ggtitle(paste0(AV_title)) + 
               coord_cartesian(ylim = c(quans_85[1], quans_85[2]))

  AV_box_45 <- box_trend_monthly_cum(dt=curr_AVs_45, 
                                     p_type="box",
                                     y_lab=AV_y_lab, 
                                     tgt_col=AV_tg_col) + 
               ggtitle(paste0(AV_title)) + 
      coord_cartesian(ylim = c(quans_45[1], quans_45[2]))
  #########
  ######### difference plot
  #########
  box_title <- paste0("Difference (%) in monthly precipitation")
  
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
                         ggtitle(paste0(box_title)) +
             coord_cartesian(ylim=c(quans_85[1], quans_85[2]))

  unbias_perc_diff_45<-box_trend_monthly_cum(dt=curr_diff_45,
                                      p_type="box",
                            y_lab="differences (%)",
                              tgt_col="perc_diff") + 
                         ggtitle(paste0(box_title)) + 
             coord_cartesian(ylim=c(quans_45[1], quans_45[2]))
  #########
  ######### rain fractions
  #########
  box_title <- paste0("Proportion (%) of monthly", 
                      " precipitation in rain form")

  quans_85 <- find_quantiles(curr_AVs_85, 
                             tgt_col="rain_fraction", 
                             time_type="monthly")

  quans_45 <- find_quantiles(curr_AVs_45, 
                             tgt_col="rain_fraction",
                             time_type="monthly")
             
  rain_frac_85 <- box_trend_monthly_cum(dt = curr_AVs_85, 
                                        p_type="box",
                                        y_lab="rain proportion (%)", 
                                        tgt_col="rain_fraction") + 
                  ggtitle(paste0(box_title)) + 
               coord_cartesian(ylim = c(max(-2, quans_85[1]), 
               	                        min(105, quans_85[2])))
  
  rain_frac_45 <- box_trend_monthly_cum(dt=curr_AVs_45,
                                        p_type="box",
                                        y_lab="rain proportion (%)", 
                                        tgt_col="rain_fraction") + 
                  ggtitle(paste0(box_title)) + 
                  coord_cartesian(ylim = c(max(-2, quans_45[1]), 
                                           min(105, quans_45[2])))
  ######################################################
  #######################################################
  #####
  ##### 3 in 1
  #####
  #######################################################
  three_in_1_dir <- paste0(plot_base)
  if (dir.exists(three_in_1_dir) == F) {
    dir.create(path = three_in_1_dir, recursive = T)}
  print (three_in_1_dir)
  three_85 <- ggarrange(plotlist = list(AV_box_85, 
                                        unbias_perc_diff_85, 
                                        rain_frac_85),
                       ncol=1, nrow=3, common.legend = TRUE, 
                       legend="bottom")

  three_85 <- annotate_figure(three_85,
                             top = text_grob(paste0(clust_g, ", RCP 8.5"), 
                                              face = "bold", 
                                              size = 10,
                                              color="red"))
  
  three_45 <- ggarrange(plotlist = list(AV_box_45, 
                                        unbias_perc_diff_45, 
                                        rain_frac_45),
                       ncol=1, nrow=3, common.legend = TRUE, 
                       legend="bottom")

  three_45 <- annotate_figure(three_45,
                              top = text_grob(paste0(clust_g, ", RCP 4.5"), 
                                              face = "bold", 
                                              size = 10,
                                              color="red"))
  
  ggsave(filename = paste0(gsub("\ ", "_", clust_g), "_85.png"),
         plot = three_85, 
         width=8, height=5, units = "in", 
         dpi=600, device = "png", path = three_in_1_dir)
  
  ggsave(filename = paste0(gsub("\ ", "_", clust_g), "_45.png"),
         plot = three_45,
         width=8, height=5, units = "in", 
         dpi=600, device = "png", path=three_in_1_dir)

}

print (Sys.time() - start_time)


