# rm(list=ls())

.libPaths("/data/hydro/R_libs35")
.libPaths()

library(data.table)
library(dplyr)
library(ggmap)
library(ggplot2)
library(ggpubr) # for ggarrange
options(digit=9)
options(digits=9)

# source_path_1 = "/home/hnoorazar/chilling_codes/current_draft/chill_core.R"
# source(source_path_1)
# source_path_2 = "/home/hnoorazar/chilling_codes/current_draft/chill_plot_core.R"
# source(source_path_2)

produce_data_4_safe_chill_box_plots_new_seasons <- function(data){

  ################### CLEAN DATA
  data <- organize_non_over_time_period_two_hist(data)

  ################### GENERATE STATS
  #######################################################################
  ##                                                                   ##
  ##   Find the 90th percentile of the chill units                     ##
  ##   Grouped by location, model, time_period and rcp                 ##
  ##   This could be used for box plots, later compute the mean.       ##
  ##   for maps                                                        ##
  ##                                                                   ##
  #######################################################################
  quan_per_loc_period_model_jan <- data %>% 
                                   group_by(time_period, lat, long, scenario, model, start) %>%
                                   summarise(quan_90 = quantile(sum_J1, probs = 0.1)) %>%
                                   data.table()
  
  quan_per_loc_period_model_feb <- data %>% 
                                   group_by(time_period, lat, long, scenario, model, start) %>%
                                   summarise(quan_90 = quantile(sum_F1, probs = 0.1)) %>%
                                   data.table()

  quan_per_loc_period_model_mar <- data %>% 
                                   group_by(time_period, lat, long, scenario, model, start) %>%
                                   summarise(quan_90 = quantile(sum_M1, probs = 0.1)) %>%
                                   data.table()

  quan_per_loc_period_model_apr <- data %>% 
                                   group_by(time_period, lat, long, scenario, model, start) %>%
                                   summarise(quan_90 = quantile(sum_A1, probs = 0.1)) %>%
                                   data.table()

  # it seems there is a library, perhaps tidyverse, that messes up
  # the above line, so the two variables above are 1-by-1. 
  # just close and re-open R Studio
  ########################################################################
  #######                                                          #######
  #######                     Means                              #######
  #######                                                          #######
  ########################################################################
  
  mean_quan_per_loc_period_model_jan <- quan_per_loc_period_model_jan %>%
                                        group_by(time_period, lat, long, scenario, start) %>%
                                        summarise(mean_over_model = mean(quan_90)) %>%
                                        data.table()

  mean_quan_per_loc_period_model_feb <- quan_per_loc_period_model_feb %>%
                                        group_by(time_period, lat, long, scenario, start) %>%
                                        summarise(mean_over_model = mean(quan_90)) %>%
                                        data.table()
  
  mean_quan_per_loc_period_model_mar <- quan_per_loc_period_model_mar %>%
                                        group_by(time_period, lat, long, scenario, start) %>%
                                        summarise(mean_over_model = mean(quan_90)) %>%
                                        data.table()

  mean_quan_per_loc_period_model_apr <- quan_per_loc_period_model_apr %>%
                                        group_by(time_period, lat, long, scenario, start) %>%
                                        summarise(mean_over_model = mean(quan_90)) %>%
                                        data.table()
  ########################################################################
  #######                                                          #######
  #######                     Medians                              #######
  #######                                                          #######
  ########################################################################
  median_quan_per_loc_period_model_jan <- quan_per_loc_period_model_jan %>%
                                          group_by(time_period, lat, long, scenario, start) %>%
                                          summarise(median_over_model = median(quan_90)) %>%
                                          data.table()

  median_quan_per_loc_period_model_feb <- quan_per_loc_period_model_feb %>%
                                          group_by(time_period, lat, long, scenario, start) %>%
                                          summarise(median_over_model = median(quan_90)) %>%
                                          data.table()
  
  median_quan_per_loc_period_model_mar <- quan_per_loc_period_model_mar %>%
                                          group_by(time_period, lat, long, scenario, start) %>%
                                          summarise(median_over_model = median(quan_90)) %>%
                                          data.table()
  
  median_quan_per_loc_period_model_apr <- quan_per_loc_period_model_apr %>%
                                          group_by(time_period, lat, long, scenario, start) %>%
                                          summarise(median_over_model = median(quan_90)) %>%
                                          data.table()

  return(list(quan_per_loc_period_model_jan,
              mean_quan_per_loc_period_model_jan,
              median_quan_per_loc_period_model_jan,
              quan_per_loc_period_model_feb,
              mean_quan_per_loc_period_model_feb,
              median_quan_per_loc_period_model_feb,
              quan_per_loc_period_model_mar,
              mean_quan_per_loc_period_model_mar,
              median_quan_per_loc_period_model_mar,
              quan_per_loc_period_model_apr,
              mean_quan_per_loc_period_model_apr,
              median_quan_per_loc_period_model_apr
              )
        )
}

organize_non_over_time_period_two_hist <- function(data){
  data = data %>% filter(year<=2005 | year>=2025)
  time_periods = c("Historical","2025_2050", "2051_2075", "2076_2099")
  
  data$time_period = 0L
  data$time_period[data$year <= 2005] = time_periods[1]
  data$time_period[data$year >= 2025 & data$year <= 2050] = time_periods[2]
  data$time_period[data$year >= 2051 & data$year <= 2075] = time_periods[3]
  data$time_period[data$year >= 2076] = time_periods[4]
  
  data$time_period = factor(data$time_period, levels = time_periods, order=T)
  data_f <- data %>% filter(time_period != "Historical")
  
  data_h_rcp85 <- data %>% filter(time_period == "Historical")
  data_h_rcp45 <- data %>% filter(time_period == "Historical")
  
  data_h_rcp85$scenario = "RCP 8.5"
  data_h_rcp45$scenario = "RCP 4.5"
  
  data_f$scenario[data_f$scenario=="rcp85"] = "RCP 8.5"
  data_f$scenario[data_f$scenario=="rcp45"] = "RCP 4.5"

  data = rbind(data_f, data_h_rcp45, data_h_rcp85)
  rm(data_h_rcp45, data_h_rcp85, data_f)
  return(data)
}

pick_single_cities <- function(dt, city_info){
  city_info$location <- paste0(city_info$lat, "_", city_info$long)
  dt$location <- paste0(dt$lat, "_", dt$long)

  dt <- dt %>% filter(location %in% city_info$location) %>%
        data.table()
  return(dt)
}

#################
## Plotting functions are in safe_chill_historicalRCPs.R
## or other files in : chill_plot_core.R
## 
#################

# setwd("/Users/hn/Desktop/Desktop/Kirti/check_point/chilling/non_overlapping/different_chill_start/")
# files <- dir(pattern = ".rds")

# for (file in files){
#   datas = data.table(readRDS(file))
#   datas <- datas %>% filter(model != "observed")
#   chill_season_start = unlist(strsplit(file, split='[.]'))[1]
#   if (chill_season_start == )

#   datas <- pick_up_okanagan_rich(datas) # Pick up okanagan And Richland
#   datas <- within(datas, remove(thresh_20, thresh_25, thresh_30, thresh_35,
#                                 thresh_40, thresh_45, thresh_50, thresh_55, 
#                                 thresh_60, thresh_65,
#                                 thresh_70, thresh_75, sum))
#   information = produce_data_4_plots(datas)

#   safe_jan <- safe_box_plot(information[[1]], due="Jan.", chill_start = chill_season_start)
#   safe_feb <- safe_box_plot(information[[4]], due="Feb.", chill_start = chill_season_start)
#   safe_mar <- safe_box_plot(information[[7]], due="Mar.", chill_start = chill_season_start)

#   # out_dir
#   output_name = paste0("start_", chill_season_start, "_Jan_1.png")
#   ggsave(output_name, safe_jan, path=plot_dir, width=7, height=7, unit="in", dpi=300)

#   output_name = paste0("start_", chill_season_start, "_Feb_1.png")
#   ggsave(output_name, safe_feb, path=plot_dir, width=7, height=7, unit="in", dpi=300)

#   output_name = paste0("start_", chill_season_start, "_Mar_1.png")
#   ggsave(output_name, safe_mar, path=plot_dir, width=7, height=7, unit="in", dpi=300)
# }

#########################################################
#######                                           #######
#######      Above plots, combined into 1         #######
#######                                           #######
#########################################################

# setwd("/Users/hn/Desktop/Desktop/Kirti/check_point/chilling/")
setwd("/data/hydro/users/Hossein/chill/data_by_core/dynamic/02")
plot_dir <- "/data/hydro/users/Hossein/chill/data_by_core/dynamic/02/safe_chill_plots/"
param_dir <- "/home/hnoorazar/chilling_codes/parameters/"

limited_locs <- read.csv(paste0(param_dir, "limited_locations.csv"), 
                                header=T, sep=",", as.is=T)

#####################################################################################

# read and clean data

dt <- readRDS("all_summary_comps.rds") %>% data.table()
dt <- dt %>% filter(model != "observed")
dt <- within(dt, remove(thresh_20, thresh_25, thresh_30, thresh_35,
                        thresh_40, thresh_45, thresh_50, thresh_55, 
                        thresh_60, thresh_65,
                        thresh_70, thresh_75, sum))

dt <- pick_single_cities(dt, limited_locs)

#####################################################################################

information = produce_data_4_safe_chill_box_plots_new_seasons(dt)

safe_jan <- information[[1]]
safe_feb <- information[[4]]
safe_mar <- information[[7]]
safe_apr <- information[[10]]

safe_jan$up_to <- "jan1"
safe_feb$up_to <- "feb1"
safe_mar$up_to <- "mar1"
safe_apr$up_to <- "apr1"

all_seasons_safe <- rbind(safe_jan, safe_feb, safe_mar, safe_apr)
all_seasons_safe <- left_join(all_seasons_safe, limited_locs)

starts <- c("Sept. 1", "Sept. 15", "Oct. 1", "Oct. 15", "Nov. 1", "Nov. 15")
all_seasons_safe$start <- factor(all_seasons_safe$start, levels = starts, order=T)

all_seasons_safe <- within(all_seasons_safe, remove(lat, long))
all_seasons_safe_dc <- dcast(all_seasons_safe,
                             time_period + scenario + start + city + model ~ up_to,
                             value.var = "quan_90")

the_theme <- theme(plot.margin = unit(c(t=.2, r=.2, b=.2, l=0.2), "cm"),
                   plot.title = element_text(size = 30, face = "bold"),
                   panel.border = element_rect(fill=NA, size=.3),
                   panel.grid.major = element_line(size = 0.05),
                   panel.grid.minor = element_blank(),
                   panel.spacing.y = unit(.35, "cm"),
                   panel.spacing.x = unit(.25, "cm"),
                   legend.position = "bottom", 
                   legend.key.size = unit(3, "line"),
                   legend.spacing.x = unit(.2, 'cm'),
                   legend.title=element_text(size = 30),
                   legend.text = element_text(size = 25),
                   legend.margin = margin(t=0, r=0, b=0, l=0, unit = 'cm'),
                   strip.text.x = element_text(size=22, face="bold", color="black"),
                   strip.text.y = element_text(size=22, face="bold", color="black"),
                   axis.ticks = element_line(size=.1, color="black"),
                   axis.text.x = element_text(size=20, face="bold", color="black", angle=-30),
                   axis.text.y = element_text(size=20, face="bold", color="black"),
                   axis.title.x = element_text(size=22, face="bold", margin = margin(t=10, r=0, b=0, l=0)),
                   axis.title.y = element_text(size=25, face="bold", margin = margin(t=0, r=10, b=0, l=0))
                   )

noch <- FALSE
box_width <- 1
color_ord = c("black", "grey47" , "dodgerblue", "olivedrab4", "red", "yellow") # 
time_lab = c("Hist.", "'25-'50", "'51-'75", "'76-'99")

title_s <- "Safe chill by Jan. 1"
jan_1_box <- ggplot(data = all_seasons_safe_dc, aes(x=time_period, y=jan1, fill=start)) +
             geom_boxplot(outlier.size = -.3, notch= noch, width=box_width, lwd=.1) +
             facet_grid(~ scenario ~  city) + 
             labs(x = "", y = "safe chill") + 
             scale_fill_manual(name = "chill season start",
                               values = color_ord,
                               labels = ) +
             scale_x_discrete(#breaks = x_breaks,
                              labels = time_lab) +
             ggtitle(title_s) +
             the_theme

title_s <- "Safe chill by Feb. 1"
feb_1_box <- ggplot(data = all_seasons_safe_dc, aes(x=time_period, y=feb1, fill=start)) +
             geom_boxplot(outlier.size = -.3, notch= noch, width=box_width, lwd=.1) +
             facet_grid(~ scenario ~  city) + 
             labs(x = "", y = "") + 
             scale_fill_manual(name = "chill season start",
                               values = color_ord,
                               labels = c("Sept. 15", "Oct. 1", "Oct. 15", "Nov. 1", "Nov. 15")) +
             scale_x_discrete(#breaks = x_breaks,
                              labels = time_lab) +
             ggtitle(title_s) +
             the_theme

title_s <- "Safe chill by Mar. 1"
mar_1_box <- ggplot(data = all_seasons_safe_dc, aes(x=time_period, y=mar1, fill=start)) +
             geom_boxplot(outlier.size = -.3, notch= noch, width=box_width, lwd=.1) +
             facet_grid(~ scenario ~  city) + 
             labs(x = "", y = "safe chill") + 
             scale_fill_manual(name = "chill season start",
                               values = color_ord,
                               labels = c("Sept. 15", "Oct. 1", "Oct. 15", "Nov. 1", "Nov. 15")) +
             scale_x_discrete(#breaks = x_breaks,
                              labels = time_lab) +
             ggtitle(title_s) +
             the_theme

title_s <- "Safe chill by Apr. 1"
apr_1_box <- ggplot(data = all_seasons_safe_dc, aes(x=time_period, y=apr1, fill=start)) +
             geom_boxplot(outlier.size = -.3, notch= noch, width=box_width, lwd=.1) +
             facet_grid(~ scenario ~ city) + 
             labs(x = "", y = "") + 
             scale_fill_manual(name = "chill season start",
                               values = color_ord,
                               labels = c("Sept. 15", "Oct. 1", "Oct. 15", "Nov. 1", "Nov. 15")) +
             scale_x_discrete(#breaks = x_breaks,
                              labels = time_lab) +
             ggtitle(title_s) +
             the_theme

all_boxes <- ggarrange(jan_1_box, feb_1_box,
                       mar_1_box, apr_1_box,
                       ncol = 2, nrow = 2, common.legend = T,
                       legend = "bottom")

ggsave("all_chills_limited_locs.png", all_boxes,
       path=plot_dir, width=45, height=15, unit="in", dpi=250)

#########################################################
#
#      Box plot of CPs
#
#########################################################

dt$start <- factor(dt$start, levels = starts)

dt <- organize_non_over_time_period_two_hist(dt)
dt <- within(dt, remove(year, location, chill_season))
dt <- left_join(dt, limited_locs)

title_s <- "Accumulated CP by Jan. 1"
jan_1_box <- ggplot(data = dt, aes(x=time_period, y=sum_J1, fill=start)) +
             geom_boxplot(outlier.size = -.3, notch= noch, width=box_width, lwd=.1) +
             facet_grid(~ scenario ~  city) + 
             labs(x = "", y = "accumulated CP") + 
             scale_fill_manual(name = "chill season start",
                               values = color_ord,
                               labels = starts) +
             scale_x_discrete(labels = time_lab) +
             ggtitle(title_s) +
             the_theme

title_s <- "Accumulated CP by Feb. 1"
feb_1_box <- ggplot(data = dt, aes(x=time_period, y=sum_F1, fill=start)) +
             geom_boxplot(outlier.size = -.3, notch= noch, width=box_width, lwd=.1) +
             facet_grid(~ scenario ~  city) + 
             labs(x = "", y = "") + 
             scale_fill_manual(name = "chill season start",
                               values = color_ord,
                               labels = starts) +
             scale_x_discrete(labels = time_lab) +
             ggtitle(title_s) +
             the_theme

title_s <- "Accumulated CP by Mar. 1"
mar_1_box <- ggplot(data = dt, aes(x=time_period, y=sum_M1, fill=start)) +
             geom_boxplot(outlier.size = -.3, notch= noch, width=box_width, lwd=.1) +
             facet_grid(~ scenario ~  city) + 
             labs(x = "", y = "accumulated CP") + 
             scale_fill_manual(name = "chill season start",
                               values = color_ord,
                               labels = c("Sept. 15", "Oct. 1", "Oct. 15", "Nov. 1", "Nov. 15")) +
             scale_x_discrete(labels = time_lab) +
             ggtitle(title_s) +
             the_theme

title_s <- "Accumulated CP by Apr. 1"
apr_1_box <- ggplot(data = dt, aes(x=time_period, y=sum_A1, fill=start)) +
             geom_boxplot(outlier.size = -.3, notch= noch, width=box_width, lwd=.1) +
             facet_grid(~ scenario ~ city) + 
             labs(x = "", y = "") + 
             scale_fill_manual(name = "chill season start",
                               values = color_ord,
                               labels = starts) +
             scale_x_discrete(labels = time_lab) +
             ggtitle(title_s) +
             the_theme

####### Arrange all box plots

all_boxes <- ggarrange(jan_1_box, feb_1_box,
                       mar_1_box, apr_1_box,
                       ncol = 2, nrow = 2, common.legend = T,
                       legend = "bottom")
ggsave("all_accum_CP_boxes_limited_locs.png", all_boxes,
       path = plot_dir, width=45, height=15, unit="in", dpi=250)



