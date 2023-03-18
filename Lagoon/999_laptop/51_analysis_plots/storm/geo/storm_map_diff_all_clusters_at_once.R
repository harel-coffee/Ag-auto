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
all_storms <- get_ridof_canada(all_storms)
head(all_storms, 2)

all_storms <- all_storms %>%
              filter(return_period != "2006-2025")%>%
              data.table()

all_storms <- within(all_storms, 
                    remove(five_years, ten_years, 
                           fifteen_years, twenty_years))

all_storms <- convert_5_numeric_clusts_to_alphabet(data_tb=all_storms)
emissions <- sort(unique(all_storms$emission))
########################################################################
####                   
####     WE DO CLUSTERS SEPARATELY SO COLORs are VISIBLE!
####
########################################################################

# biased_dt <- storm_diff_obs_or_modeled(dt_dt=all_storms, 
#                                        diff_from="1979-2016")
# biased_dt <- biased_dt %>%
#              group_by(location, emission, return_period, cluster) %>% 
#              summarise(perc_diff_meds = median(perc_diff)) %>% 
#              data.table()

unbiased_dt <- storm_diff_obs_or_modeled(dt_dt=all_storms, 
                                         diff_from="1950-2005")

unbiased_dt <- unbiased_dt %>%
               group_by(location, emission, return_period, cluster) %>% 
               summarise(perc_diff_meds = median(perc_diff)) %>% 
               data.table()
time_ps <- sort(unique(biased_dt$return_period))
tp <- time_ps[1]
em <- emissions[1]

unbiased_min <- min(unbiased_dt$perc_diff_meds)
unbiased_max <- max(unbiased_dt$perc_diff_meds)

for (em in emissions){
  for (tp in time_ps){
    # plt_dt_bias <- biased_dt %>% 
    #                filter(emission == em & return_period==tp) %>% 
    #                data.table()

    plt_dt_unbias <- unbiased_dt %>% 
                     filter(emission == em & return_period==tp) %>% 
                     data.table()

    # biased_min <- min(plt_dt_bias$perc_diff_meds)
    # biased_max <- max(plt_dt_bias$perc_diff_meds)

    # unbiased_min <- min(plt_dt_unbias$perc_diff_meds)
    # unbiased_max <- max(plt_dt_unbias$perc_diff_meds)

    # bias_clr_lim <- c(floor(biased_min), ceiling(biased_max))
    unbias_clr_lim <- c(floor(unbiased_min), ceiling(unbiased_max))

    # assign(x=paste0("bias_", 
    #                 gsub("[.]", "", gsub("\ ", "", tolower(em))),
    #                 "_", gsub("-", "_", tp)),
    #        value = geo_map_perc_diff(dt_dt = plt_dt_bias,
    #                                  col_col = "perc_diff_meds", 
    #                                  color_limit = bias_clr_lim) + 
    #                ggtitle(label=paste0(em, 
    #                                     "- 25-year/24-hour storm design", 
    #                                     "(", tp, ")")))
                                      
    assign(x = paste0("unbias_", 
                      gsub("[.]", "", gsub("\ ", "", tolower(em))),
                      "_", gsub("-", "_", tp)),
           value = geo_map_perc_diff(dt_dt = plt_dt_unbias,
                                     col_col = "perc_diff_meds", 
                                     color_limit = unbias_clr_lim) + 
                   ggtitle(label=paste0("25-year/24-hour design storm",
                                        "\n", "percent difference from historical", 
                                        "\n", em, ", ", tp)))
  }
}

unbias_45 <- ggarrange(plotlist = list(unbias_rcp45_2026_2050,
                                       unbias_rcp45_2051_2075,
                                       unbias_rcp45_2076_2099),
                         ncol=3, nrow=1, common.legend=FALSE)

unbias_85 <- ggarrange(plotlist = list(unbias_rcp85_2026_2050,
                                       unbias_rcp85_2051_2075,
                                       unbias_rcp85_2076_2099),
                         ncol=3, nrow=1, common.legend=FALSE)

ggsave(filename = paste0("unbias_45_all_clust_16_inch_wide.png"), 
       plot=unbias_45, 
       width=16, height=4, units="in", 
       dpi=600, device="png", path=diff_plot_dir)

ggsave(filename = paste0("unbias_85_all_clust_16_inch_wide.png"), 
       plot=unbias_85, 
       width=16, height=4, units="in", 
       dpi=600, device="png", path=diff_plot_dir)

ggsave(filename = paste0("unbias_45_all_clust_14_inch_wide.png"), 
       plot=unbias_45, 
       width=14, height=4, units="in", 
       dpi=600, device="png", path=diff_plot_dir)

ggsave(filename = paste0("unbias_85_all_clust_14_inch_wide.png"), 
       plot=unbias_85, 
       width=14, height=4, units="in", 
       dpi=600, device="png", path=diff_plot_dir)

ggsave(filename = paste0("unbias_45_all_clust_12_inch_wide.png"), 
       plot=unbias_45, 
       width=12, height=4, units="in", 
       dpi=600, device="png", path=diff_plot_dir)

ggsave(filename = paste0("unbias_85_all_clust_12_inch_wide.png"), 
       plot=unbias_85, 
       width=12, height=4, units="in", 
       dpi=600, device="png", path=diff_plot_dir)