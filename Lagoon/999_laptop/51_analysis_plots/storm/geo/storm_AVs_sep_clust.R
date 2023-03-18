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
nd_cols <- c("location", "model", "emission", 
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
                   group_by(location, emission, return_period) %>% 
                   summarise(twenty_five_years = median(twenty_five_years)) %>% 
                   data.table()

storm_mod_hist <- storm_mod_hist %>%
                  group_by(location, return_period) %>% 
                  summarise(twenty_five_years = median(twenty_five_years)) %>% 
                  data.table()

storm_F_medians <- merge(storm_F_medians, fip_clust, all.x=TRUE)
storm_mod_hist <- merge(storm_mod_hist, fip_clust, all.x=TRUE)
clusters <- unique(storm_F_medians$cluster)

clust <- clusters[1]

for (clust in clusters){
  curr_hist <- storm_mod_hist %>% filter(cluster == clust)
  curr_ft <- storm_F_medians %>% filter(cluster == clust)

  minnn <- min(curr_hist$twenty_five_years)
  maxxx <- max(curr_hist$twenty_five_years)

  minnn <- min(minnn, min(curr_ft$twenty_five_years))
  maxxx <- max(maxxx, max(curr_ft$twenty_five_years))

  assign(x = paste0("model_hist_plt"),
        value = obs_hist_map_storm(dt=curr_hist, 
                               minn=minnn, maxx=maxxx, 
                               fips_clust=fip_clust,
                               tgt_col="twenty_five_years") + 
                ggtitle("Modeled historical"))

  emissions <- c("RCP 4.5", "RCP 8.5")
  em <- emissions[1]; rp <- future_rn_pr[1]

  for (em in emissions){
    for (rp in future_rn_pr){
      curr_dt <- curr_ft %>%
                 filter(emission == em & return_period==rp) %>%
                 data.table()
      title <- paste0(em, " (", rp, ")")
      assign(x = paste0(gsub("[.]", "", gsub("\ ", "", tolower(em))), "_",
                        gsub(pattern = "-", replacement = "_", x = rp)),
             value ={one_time_medians_storm_geoMap(curr_dt, 
                                                   minn=minnn, maxx=maxxx, 
                                                   ttl=title, subttl=subttl, 
                                                   differ=FALSE)})
    }
  }
  assign(x = paste0("median_45_", gsub("\ ", "_", clust)),
         value = ggarrange(plotlist = list(model_hist_plt, 
                                           rcp45_2026_2050,
                                           rcp45_2051_2075, 
                                           rcp45_2076_2099),
                           ncol=4, nrow=1, common.legend=FALSE))

  assign(x = paste0("median_85_", gsub("\ ", "_", clust)),
         value = ggarrange(plotlist = list(model_hist_plt,
                                           rcp85_2026_2050,
                                           rcp85_2051_2075,
                                           rcp85_2076_2099),
                           ncol=4, nrow=1, common.legend=FALSE))
}

AV_85 <- ggarrange(plotlist = list(median_85_Western_coastal,
                                   median_85_Cascade_foothills,
                                   median_85_Northwest_Cascades,
                                   median_85_Northcentral_Cascades,
                                   median_85_Northeast_Cascades),
                     ncol=1, nrow=5, common.legend=TRUE, 
                     legend="bottom")

AV_45 <- ggarrange(plotlist = list(median_45_Western_coastal,
                                   median_45_Cascade_foothills,
                                   median_45_Northwest_Cascades,
                                   median_45_Northcentral_Cascades,
                                   median_45_Northeast_Cascades),
                     ncol=1, nrow=5, common.legend=FALSE)

AV_85 <- annotate_figure(AV_85,
                         top = text_grob(" ", 
                                         color = "red", face = "bold", 
                                         size = 14),
                         fig.lab="RCP 8.5 - 25-year/24-hour", 
                         fig.lab.face = "bold", fig.lab.size=14)

AV_45 <- annotate_figure(AV_45,
                         top = text_grob(" ", 
                                         color = "red", face = "bold", 
                                         size = 14),
                         fig.lab="RCP 4.5 - 25-year/24-hour", 
                         fig.lab.face = "bold", fig.lab.size=14)

ggsave(filename = paste0("AV_85_sep_clust.png"), 
       plot=AV_85, 
       width=8, height=10, units="in", 
       dpi=600, device="png", path=AVs_plot_dir)

ggsave(filename = paste0("AV_45_sep_clust.png"), 
       plot=AV_45, 
       width=8, height=10, units="in", 
       dpi=600, device="png", path=AVs_plot_dir)




