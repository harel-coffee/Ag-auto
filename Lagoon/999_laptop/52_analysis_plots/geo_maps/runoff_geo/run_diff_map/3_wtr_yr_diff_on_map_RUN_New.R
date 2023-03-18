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

in_dir <- "/Users/hn/Desktop/Desktop/Ag/check_point/lagoon/runbase/"
diff_plot_dir <- paste0(in_dir, "narrowed_runbase/maps/diffs/")
if(dir.exists(diff_plot_dir) == F){dir.create(path=diff_plot_dir,
                                          recursive = T)}
##############################

fileN <- "wtr_yr_cum_runbase"
dt_tb <- data.table(readRDS(paste0(in_dir, fileN, ".rds")))
head(dt_tb, 2)
dim(dt_tb)
length(unique(dt_tb$location))

dt_tb <- convert_5_numeric_clusts_to_alphabet(data_tb=dt_tb)
cluster_inf <- subset(dt_tb, select=c(location, cluster))
tgt_col <- "annual_cum_runbase"
meds <- median_diff_obs_or_modeled(dt_tb, 
                                   tgt_col=tgt_col, 
                                   diff_from="1950-2005")
meds <- median_of_diff_of_medians(meds)

clusters <- unique(meds$cluster)
emissions <- c("RCP 4.5", "RCP 8.5")
future_rn_pr <- c("2026-2050", "2051-2075", "2076-2099")

#######
#######     Percentage perc_difference of medians of annual precip
#######
em <- emissions[1]; rp <- future_rn_pr[1]; clust <- clusters[1]
subtitle <- "Diff. of medians\nof cum. runoff (Water Year)"

for (clust in clusters){
  curr_dt_clust <- meds %>% filter(cluster == clust)
  for (em in emissions){
    for (rp in future_rn_pr){
      curr_dt <- curr_dt_clust %>%
                 filter(emission == em & time_period==rp) %>%
                 data.table()
      min_diff_perc <- min(curr_dt$perc_med_of_diffs_of_meds)
      max_diff_perc <- max(curr_dt$perc_med_of_diffs_of_meds)
  
      title <- paste0(em, " (", rp, ")")
      
      assign(x = paste0(gsub("[.]", "", gsub("\ ", "", tolower(em))), "_",
                        gsub(pattern = "-", replacement = "_", x = rp)),
             value ={geo_map_of_diffs(dt = curr_dt, 
                                      col_col = "perc_med_of_diffs_of_meds",
                                      minn=min_diff_perc, 
                                      maxx=max_diff_perc,
                                      ttl = title, 
                                      subttl= subtitle)})
    }
  }
  assign(x = paste0("perc_diff_figs_45_", gsub("\ ", "_", clust)),
         value = ggarrange(plotlist = list(rcp45_2026_2050,
                                           rcp45_2051_2075,
                                           rcp45_2076_2099),
                           ncol=3, nrow=1, common.legend=FALSE))

  assign(x = paste0("perc_diff_figs_85_", gsub("\ ", "_", clust)),
         value = ggarrange(plotlist = list(rcp85_2026_2050,
                                           rcp85_2051_2075,
                                           rcp85_2076_2099),
                           ncol=3, nrow=1, common.legend=FALSE))
}

perc_diff_85<-ggarrange(plotlist=list(perc_diff_figs_85_Western_coastal,
                        perc_diff_figs_85_Cascade_foothills,
                        perc_diff_figs_85_Northwest_Cascades,
                        perc_diff_figs_85_Northcentral_Cascades,
                        perc_diff_figs_85_Northeast_Cascades),
                        ncol=1, nrow=5, common.legend=FALSE)

perc_diff_45<-ggarrange(plotlist = list(perc_diff_figs_45_Western_coastal,
                        perc_diff_figs_45_Cascade_foothills,
                        perc_diff_figs_45_Northwest_Cascades,
                        perc_diff_figs_45_Northcentral_Cascades,
                        perc_diff_figs_45_Northeast_Cascades),
                        ncol=1, nrow=5, common.legend=FALSE)

ggsave(filename = paste0("perc_diff_85_sep_clust.png"), 
       plot=perc_diff_85, 
       width=11, height=15, units="in", 
       dpi=600, device="png", path=diff_plot_dir)

ggsave(filename = paste0("perc_diff_45_sep_clust.png"), 
       plot=perc_diff_45, 
       width=11, height=15, units="in", 
       dpi=600, device="png", path=diff_plot_dir)

rm(perc_diff_figs_85_Western_coastal,
   perc_diff_figs_85_Cascade_foothills,
   perc_diff_figs_85_Northwest_Cascades,
   perc_diff_figs_85_Northcentral_Cascades,
   perc_diff_figs_85_Northeast_Cascades,
   perc_diff_figs_45_Western_coastal,
   perc_diff_figs_45_Cascade_foothills,
   perc_diff_figs_45_Northwest_Cascades,
   perc_diff_figs_45_Northcentral_Cascades,
   perc_diff_figs_45_Northeast_Cascades)

#######
#######     Difference of medians of annual precip
#######
# min_diff <- min(meds$diff)
# max_diff <- max(meds$diff)

# subtitle <- "Diff. of medians\nof cum. runoff (Water Year)"
# for (em in emissions){
#   for (rp in future_rn_pr){
#     curr_dt <- meds %>%
#                filter(emission == em & time_period==rp) %>%
#                data.table()
#     title <- paste0(em, " (", rp, ")")
    
#     assign(x = paste0(gsub(pattern = " ", 
#                            replacement = "_", 
#                            x = em),
#                       "_",
#                       gsub(pattern = "-", 
#                            replacement = "_", 
#                            x = rp)),
#            value ={geo_map_of_diffs(dt = curr_dt, col_col = "diff" , 
#                                     minn = min_diff, maxx = max_diff,
#                                     ttl = title, subttl= subtitle)})

#   }
# }

# diff_figs <- ggarrange(plotlist = list(RCP_8.5_2026_2050,
#                                        RCP_8.5_2051_2075,
#                                        RCP_8.5_2076_2099,
#                                        RCP_4.5_2026_2050,
#                                        RCP_4.5_2051_2075,
#                                        RCP_4.5_2076_2099),
#                        ncol = 3, nrow = 2,
#                        common.legend = TRUE)

# ggsave(filename = "run_diff_medians_WTRYR.png", 
#        plot = diff_figs, 
#        width = 10, height = 7, units = "in", 
#        dpi=300, device = "png",
#        path = plot_dir)

# rm(RCP_4.5_2026_2050, RCP_8.5_2026_2050,
#    RCP_4.5_2051_2075, RCP_8.5_2051_2075,
#    RCP_4.5_2076_2099, RCP_8.5_2076_2099, diff_figs)

