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
########################################
start_time <- Sys.time()
base <- "/Users/hn/Desktop/Desktop/Ag/check_point/lagoon/"
data_base <- paste0(base, "rain_snow_fractions/")
AV_fileNs <- c("wtr_yr_fracs") # "annual_fracs",
timeP_ty_middN <- c("wtr_yr") # "ann", 
timeP_ty <- 1

##########################################
#
# unbias diff data directory
#
diff_dir <- paste0(base, "precip/02_med_diff_med_no_bias/")

AV_y_lab <- "precipitation (mm)"
AV_title <- paste0("Total annual precipitation")
AV_tg_col <- "annual_cum_precip"

for (timeP_ty in 1:1){ # annual or wtr_yr?
  
  ###############################################################
  AVs <- readRDS(paste0(data_base, AV_fileNs[timeP_ty], ".rds")) %>% 
         data.table()
  unbias_diff <- readRDS(paste0(diff_dir, "detail_med_diff_med_", 
                                timeP_ty_middN[timeP_ty], 
                                "_precip.rds")) %>% data.table()

  AVs <- subset(AVs, select = c("location", "cluster", "year", 
                                "time_period", "model",
                                "emission", "annual_cum_precip",
                                "rain_fraction", "snow_fraction"))

  AVs <- remove_observed(AVs)
  AVs <- remove_current_timeP(AVs) # remove 2006-2025
  AVs <- convert_5_numeric_clusts_to_alphabet(data_tb = AVs)

  unbias_diff <- remove_observed(unbias_diff)
  # remove 2006-2025
  unbias_diff <- remove_current_timeP(unbias_diff) 
  unbias_diff <- convert_5_numeric_clusts_to_alphabet(unbias_diff)
  ############################################################
  ############
  ############           Separate Clusters here
  ############
  ############################################################
  uniqueClusters <- unique(unbias_diff$cluster)
  unique_ems <- unique(unbias_diff$emission)
  
  for (clust in uniqueClusters){
    for (em in unique_ems){
      curr_AV <- AVs %>% 
                 filter(emission==em & cluster==clust) %>% 
                 data.table()

      curr_unbias_diff <- unbias_diff %>% 
                          filter(emission==em & cluster==clust) %>% 
                          data.table()
      ##################################
      #####
      #####    AVs plots
      #####
      ##################################
      quans <- find_quantiles(curr_AV, tgt_col= AV_tg_col, 
                              time_type="annual")

      AV_box <- ann_box_sep_cluster(dt = curr_AV, 
                                    y_lab = AV_y_lab, 
                                    tgt_col = AV_tg_col) + 
               ggtitle(AV_title) +
               coord_cartesian(ylim = c(max(0, quans[1]), quans[2]))
      ###################################
      #####
      ##### fraction plots
      #####
      ###################################
      box_title <- paste0("Proportion (%) of annual", 
                          " precipitation in rain form")

      quans <- 100 * find_quantiles(curr_AV, 
                                    tgt_col="rain_fraction", 
                                    time_type="annual")
      
      rain_frac_box <- annual_frac_sep_clust(data_tb = curr_AV,
                                             y_lab = "rain proportion (%)", 
                                             tgt_col="rain_fraction") +
                       ggtitle(box_title) + 
                       coord_cartesian(ylim = c(max(-2, quans[1]), 
                                               min(quans[2], 110)))

      ###################################
      #####
      ##### difference plots
      #####
      ###################################
      box_title <- paste0("Difference (%) in annual precipitation")

      quans <- find_quantiles(curr_unbias_diff, 
                              tgt_col= "perc_diff", 
                              time_type="annual")
    
      unbias_perc_diff <- ann_box_sep_cluster(dt=curr_unbias_diff,
                                              y_lab="differences (%)",
                                              tgt_col="perc_diff",
                                              ttl=box_title, 
                                              subttl=box_subtitle) + 
                          ggtitle(box_title) +
                          coord_cartesian(ylim = c(quans[1], quans[2]))
      ###########################################
      #####
      ##### save plots
      #####
      plot_base <- paste0(base, "plots/")
     
      ###################################
      # 
      # 3 in 1
      #
      plot_3_in_1_dir <- paste0(plot_base, "precip/", 
                                timeP_ty_middN[timeP_ty])
      if (dir.exists(plot_3_in_1_dir) == F) {
        dir.create(path = plot_3_in_1_dir, recursive = T)}
      print (plot_3_in_1_dir)

      rain <- ggarrange(plotlist = list(AV_box, 
                                        unbias_perc_diff,
                                        rain_frac_box),
                        ncol = 1, nrow = 3, 
                        common.legend = TRUE, legend="bottom")

      rain <- annotate_figure(rain,
                              top = text_grob(paste0(clust, ", ", em), 
                                              face = "bold",
                                              size = 10,
                                              color="red"))
      
      if (em == "RCP 4.5"){aaa="45"} else {aaa = "85"}
      
      ggsave(filename = paste0(str_replace(clust, " ", "_"),
                               "_", aaa, 
                               ".png"),
             plot = rain, 
             width=3.5, height=5, units = "in", 
             dpi=600, device = "png", path = plot_3_in_1_dir)

    }
  }
}
print (Sys.time() - start_time)
