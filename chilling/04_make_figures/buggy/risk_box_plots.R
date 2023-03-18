
rm(list=ls())
library(data.table)
library(dplyr)
library(tidyr)
library(tidyverse)
library(ggpubr) # for ggarrange

options(digit=9)
options(digits=9)

##########################################################################################
###                                                                                    ###
###                             Define Functions here                                  ###
###                                                                                    ###
##########################################################################################
define_path <- function(model_name){
  if (model_name == "dynamic"){
      in_dir <- paste0(main_in_dir, model_specific_dir_name[1])
      } else if (model == "utah"){
          in_dir <- paste0(main_in_dir, model_specific_dir_name[2])
  }
}

clean_process <- function(dt){
  dt <- subset(dt, select=c(chill_season, sum_J1, 
                            sum_F1, sum_M1, sum_A1,lat, long, climate_type,
                            scenario, model, year))
  
  dt <- dt %>% filter(year <= 2005 | year >= 2025)
  
  time_periods = c("Historical", "2025_2050", "2051_2075", "2076_2099")
  dt$time_period = 0L
  dt$time_period[dt$year <= 2005] <- time_periods[1]
  dt$time_period[dt$year >= 2025 & dt$year <= 2050] <- time_periods[2]
  dt$time_period[dt$year >  2050 & dt$year <= 2075] <- time_periods[3]
  dt$time_period[dt$year >  2075] <- time_periods[4]
  dt$time_period = factor(dt$time_period, levels=time_periods, order=T)

  dt$scenario[dt$scenario == "rcp45"] <- "RCP 4.5"
  dt$scenario[dt$scenario == "rcp85"] <- "RCP 8.5"
  dt$scenario[dt$scenario == "historical"] <- "Historical"

  jan_data <- subset(dt, select=c(sum_J1, lat, long, climate_type, scenario, model, time_period, chill_season))
  feb_data <- subset(dt, select=c(sum_F1, lat, long, climate_type, scenario, model, time_period, chill_season))
  mar_data <- subset(dt, select=c(sum_M1, lat, long, climate_type, scenario, model, time_period, chill_season))
  apr_data <- subset(dt, select=c(sum_A1, lat, long, climate_type, scenario, model, time_period, chill_season))
  return (list(jan_data, feb_data, mar_data, apr_data))
}

count_years_threshs_met <- function(dataT, due){
  ## This function detects the number of years
  ## for which a given location, in a given 
  ## model, e.g. BNU, in a given time_period, e.g. historical,
  ## has met a given threshold.

  # result <- data %>% 
  #           group_by(lat, long, time_period, scenario, model, climate_type) %>%
  #           summarise_all(funs(sum(. == 0))) %>% data.table()
  h_year_count <- length(unique(dataT[dataT$time_period =="Historical",]$chill_season))
  f1_year_count <- length(unique(dataT[dataT$time_period== "2025_2050",]$chill_season))
  f2_year_count <- length(unique(dataT[dataT$time_period== "2051_2075",]$chill_season))
  f3_year_count <- length(unique(dataT[dataT$time_period== "2076_2099",]$chill_season))
  if (due == "Jan"){
    col_name = "sum_J1"
    } else if (due == "Feb"){
      col_name = "sum_F1"
    } else if(due =="Mar"){
      col_name = "sum_M1"
    } else if(due =="Apr"){
      col_name = "sum_A1"
  }

  bks = c(-300, seq(20, 75, 5), 300)

  result <- dataT %>%
            mutate(thresh_range = cut(get(col_name), breaks = bks)) %>%
            group_by(lat, long, climate_type, time_period, 
                     thresh_range, model, scenario) %>%
            summarize(no_years = n_distinct(chill_season, na.rm = FALSE)) %>% 
            data.table()
  result[is.na(result)] <- 0
  
  # print(sum(is.na(result)))
  # result <- na.omit(result)

  time_periods = c("Historical", "2025_2050", "2051_2075", "2076_2099")
  result$time_period = factor(result$time_period, 
                              levels=time_periods,
                              order=T)
  ####################################################3
  # result[result$thresh_range == "(75,200]" ]$thresh_range = "75"
  # result[result$thresh_range == "(70,75]" ]$thresh_range = "70"
  # result[result$thresh_range == "(65,70]" ]$thresh_range = "65"
  # result[result$thresh_range == "(60,65]" ]$thresh_range = "60"
  # result[result$thresh_range == "(55,60]" ]$thresh_range = "55"
  # result[result$thresh_range == "(50,55]" ]$thresh_range = "50"
  # result[result$thresh_range == "(45,50]" ]$thresh_range = "45"
  # result[result$thresh_range == "(40,45]" ]$thresh_range = "40"
  # result[result$thresh_range == "(35,40]" ]$thresh_range = "35"
  # result[result$thresh_range == "(30,35]" ]$thresh_range = "30"
  # result[result$thresh_range == "(25,30]" ]$thresh_range = "25"
  # result[result$thresh_range == "(20,25]" ]$thresh_range = "20"

  # level_s = c("20", "25", "30", "35", "40", "45", "50", "55", "60", "65", "70", "75")
  # result$thresh_range = factor(result$thresh_range, levels =level_s,  order=T)
  #########################################3
  
  result$thresh_range <- factor(result$thresh_range, order=T)
  result$thresh_range <- fct_rev(result$thresh_range)
  result <- result[order(thresh_range), ]

  # df %>% group_by(id) %>% mutate(csum = cumsum(value))
  # df$csum <- ave(df$value, df$id, FUN=cumsum)
  result <- result %>% 
            group_by(lat, long, climate_type, time_period, model, scenario) %>% 
            mutate(n_years_passed = cumsum(no_years)) %>% 
            data.table()
  ########
  ######## Debuging purposes
  ########
  # first_location <- result %>% 
  #                   filter(lat == result$lat[1], 
  #                            long == result$long[1],
  #                            model == "CCSM4",
  #                            time_period == "2076_2099", 
  #                            scenario == "RCP 8.5") %>% 
  #                   data.table()

  # first_location <- first_location[order(-thresh_range, model, time_period), ]
  ##########################################################################
  #                                                                        #
  #  There are 55 years of historical                                      #
  #            24 years in 2025_2050                                       #
  #            24 years in 2051_2075                                       #
  #            23 years in 2076_2099 (why? what happened in there?)        #
  #                                                                        #
  ##########################################################################    
  # the following can be done more efficiently!
  result_hist <- result %>% filter(time_period == "Historical") %>% data.table()
  result_50 <- result %>% filter(time_period == "2025_2050") %>% data.table()
  result_75 <- result %>% filter(time_period == "2051_2075") %>% data.table()
  result_99 <- result %>% filter(time_period == "2076_2099") %>% data.table()
  
  result_hist$frac_passed = result_hist$n_years_passed / h_year_count
  result_50$frac_passed = result_50$n_years_passed / f1_year_count
  result_75$frac_passed = result_75$n_years_passed / f2_year_count
  result_99$frac_passed = result_99$n_years_passed / f3_year_count

  result <- rbind(result_hist, result_50, result_75, result_99)
  result <- na.omit(result)
  ########
  ######## Debuging purposes
  ########
  # first_location <- result %>% 
  #                   filter(lat == result$lat[1], 
  #                            long == result$long[1],
  #                            model == "CCSM4",
  #                            time_period == "2076_2099", 
  #                            scenario == "RCP 8.5") %>% 
  #                   data.table()

  # first_location <- first_location[order(-thresh_range, model, time_period), ]

  return(result)
}

plot_boxes <- function(p_data, due, noch=T, start){
  color_ord = c("grey70" , "dodgerblue", "olivedrab4", "red") # 
  time_lab = c("Historical", "2025-2050", "2051-2075", "2076-2099")
  
  box_width = 0.8
  if (due == "Jan"){
    title_s = "Thresholds met by Jan. 1st"
    } else if (due == "Feb") {
      title_s = "Thresholds met by Feb. 1st"
    } else if (due == "Mar"){
      title_s = "Thresholds met by Mar. 1st"
    } else if (due == "Apr"){
      title_s = "Thresholds met by Apr. 1st"
  }
  # reverse the order of thresholds so
  # they appear from small to large in the plot
  # We can rename them as well.
  p_data$thresh_range <- fct_rev(p_data$thresh_range)

  thresh_lab <- levels(p_data$thresh_range)
  thresh_lab <- unlist(strsplit(thresh_lab, ","))
  thresh_lab <- thresh_lab[c(TRUE, FALSE)]
  thresh_lab <- unlist(strsplit(thresh_lab, "(", fixed=T))
  thresh_lab <- thresh_lab[c(FALSE, TRUE)]

  # do the following so historical data appear in both RCP's subplots
  p_data_f <- p_data %>% filter(scenario != "Historical")
  p_data_h_45 <- p_data %>% filter(scenario == "Historical")
  p_data_h_85 <- p_data %>% filter(scenario == "Historical")
  p_data_h_45$scenario = "RCP 4.5"
  p_data_h_85$scenario = "RCP 8.5"
  p_data = rbind(p_data_h_45, p_data_h_85, p_data_f)

  the_theme <-theme_bw() + 
              theme(plot.margin = unit(c(t=.2, r=.2, b=.2, l=0.2), "cm"),
                    panel.border = element_rect(fill=NA, size=.3),
                    panel.grid.major = element_line(size = 0.05),
                    panel.grid.minor = element_blank(),
                    panel.spacing.y = unit(.35, "cm"),
                    panel.spacing.x = unit(.25, "cm"),
                    legend.position = "bottom", 
                    legend.key.size = unit(1, "line"),
                    legend.spacing.x = unit(.2, 'cm'),
                    legend.text = element_text(size=11),
                    legend.margin = margin(t=0, r=0, b=0, l=0, unit = 'cm'),
                    legend.title = element_blank(),
                    strip.text.x = element_text(size=10),
                    strip.text.y = element_text(size=10),
                    axis.ticks = element_line(size=.1, color="black"),
                    # axis.text.x = element_blank(), # element_text(size=7, face="plain", color="black"),
                    axis.text.y = element_text(size=10, face="plain", color="black"),

                    axis.title.x = element_text(size=13, face="plain", margin = margin(t=10, r=0, b=0, l=0)),
                    axis.title.y = element_text(size=13, face="plain", margin = margin(t=0, r=8, b=0, l=0))
                    )
  box <- ggplot(data = p_data, aes(x=thresh_range, y=frac_passed, fill=time_period)) +
         geom_boxplot(outlier.size = -.3, notch= noch, width=box_width, lwd=.1) +
         labs(x = "thresholds", y = "chill portion fraction met") +
         facet_grid(~ scenario ~  climate_type) + 
         scale_fill_manual(values = color_ord,
                           name = "Time\nPeriod", 
                           labels = time_lab) + 
         scale_x_discrete(#breaks = x_breaks,
                          labels = thresh_lab)  +
         ggtitle(title_s)  +
         the_theme
  output_name <- paste0(start, "_start_", due, "_thresholds.png")
  ggsave(output_name, box, 
         path=plot_path, width=10, height=4, unit="in", dpi=400)
  return(box)
}

##########################################################################################
###                                                                                    ###
###                                   Driver                                           ###
###                                                                                    ###
##########################################################################################

# main_in_dir = "/Users/hn/Desktop/Desktop/Kirti/check_point/chilling/non_overlapping/"
# model_names = c("dynamic") # , "utah"
# model_specific_dir_name = paste0(model_names, "_model_stats/")

main_in <- "/Users/hn/Desktop/Desktop/Kirti/check_point/chilling/"
setwd(main_in)

plot_path = "/Users/hn/Desktop/plots/"
starts <- c("sept", "mid_sept", "oct", "mid_oct", "nov", "mid_nov")
for (st in starts){
  file = paste0(st, "_summary_comp.rds")
  mdata <- data.table(readRDS(file))
  mdata <- mdata %>% filter(model != "observed")

  climate_type_order = c("Cooler Area", "Warmer Area", "Oregon Area")
  mdata$climate_type <- factor(mdata$climate_type, levels = climate_type_order)
  ########################################################
  #
  # Pick up Omak And Richland
  #
  ########################################################
  # datas$location = paste0(datas$lat, "_", datas$long)
  # datas <- datas %>% filter(datas$location == "48.40625_-119.53125" | datas$location == "46.28125_-119.34375")
  # datas <- within(datas, remove(location))
  # data$CountyGroup[data$location == "48.40625_-119.53125"] = "omak"
  # data$CountyGroup[data$location == "46.28125_-119.34375"] = "rich"

  information <- clean_process(mdata)
  jan_data = information[[1]]
  feb_data = information[[2]]
  mar_data = information[[3]]
  apr_data = information[[4]]
  rm(information, mdata)

  jan_result = count_years_threshs_met(dataT = jan_data, due="Jan")
  feb_result = count_years_threshs_met(dataT = feb_data, due="Feb")
  mar_result = count_years_threshs_met(dataT = mar_data, due="Mar")
  apr_result = count_years_threshs_met(dataT = apr_data, due="Apr")
  rm(jan_data, feb_data, mar_data)

  jan_plot <- plot_boxes(p_data=jan_result, due="Jan", noch=F, start=st)
  feb_plot <- plot_boxes(p_data=feb_result, due="Feb", noch=F, start=st)
  mar_plot <- plot_boxes(p_data=mar_result, due="Mar", noch=F, start=st)
  apr_plot <- plot_boxes(p_data=apr_result, due="Apr", noch=F, start=st)

  big_plot <- ggarrange(jan_plot, 
                        feb_plot,
                        mar_plot,
                        apr_plot, 
                        label.x = "threshold",
                        label.y = "chill portion fraction met",
                        ncol = 1, 
                        nrow = 4, 
                        common.legend = T,
                        legend = "bottom")
  ggsave(paste0(st, "_start_all_in_one.png"),
         big_plot,
         path=plot_path, 
         width=10, height=12, unit="in", dpi=300)
}

######################################################################################################
#######
#######    Minor modification to plot just limited cities
#######
######################################################################################################
##########################################################################################
###                                                                                    ###
###                             Define Functions here                                  ###
###                                                                                    ###
##########################################################################################

pick_single_cities <- function(dt, param_d){
  lcc <- read.table(paste0(param_d, "limited_locations.csv"), header=T, sep=",", as.is = TRUE)
  dt_local = data.table()

  for (ii in (1:dim(lcc)[1])){
    curr_dt <- dt %>% filter(lat== lcc$lat[ii] & long==lcc$long[ii])
    curr_dt$city <- lcc$city[ii]
    dt_local <- rbind(dt_local, curr_dt)
  }
  rm(dt)
  return(data.table(dt_local))
}

clean_process <- function(dt){
  dt <- subset(dt, select=c(chill_season, sum_J1, 
                            sum_F1, sum_M1, sum_A1,lat, long, # climate_type,
                            scenario, model, year, city))
  
  dt <- dt %>% filter(year <= 2005 | year >= 2025)
  
  time_periods = c("Historical", "2025_2050", "2051_2075", "2076_2099")
  
  dt$time_period[dt$year <= 2005] <- time_periods[1]
  dt$time_period[dt$year >= 2025 & dt$year <= 2050] <- time_periods[2]
  dt$time_period[dt$year >  2050 & dt$year <= 2075] <- time_periods[3]
  dt$time_period[dt$year >  2075] <- time_periods[4]
  dt$time_period = factor(dt$time_period, levels=time_periods, order=T)

  dt$scenario[dt$scenario == "rcp45"] <- "RCP 4.5"
  dt$scenario[dt$scenario == "rcp85"] <- "RCP 8.5"
  dt$scenario[dt$scenario == "historical"] <- "Historical"

  jan_data <- subset(dt, select=c(sum_J1, city, scenario, model, time_period, chill_season))
  feb_data <- subset(dt, select=c(sum_F1, city, scenario, model, time_period, chill_season))
  mar_data <- subset(dt, select=c(sum_M1, city, scenario, model, time_period, chill_season))
  apr_data <- subset(dt, select=c(sum_A1, city, scenario, model, time_period, chill_season))
  return (list(jan_data, feb_data, mar_data, apr_data))
}

count_years_threshs_met <- function(dataT, due){
  h_year_count <- length(unique(dataT[dataT$time_period =="Historical",]$chill_season))
  f1_year_count <- length(unique(dataT[dataT$time_period== "2025_2050",]$chill_season))
  f2_year_count <- length(unique(dataT[dataT$time_period== "2051_2075",]$chill_season))
  f3_year_count <- length(unique(dataT[dataT$time_period== "2076_2099",]$chill_season))
  if (due == "Jan"){
    col_name = "sum_J1"
    } else if (due == "Feb"){
      col_name = "sum_F1"
    } else if(due =="Mar"){
      col_name = "sum_M1"
    } else if(due =="Apr"){
      col_name = "sum_A1"
  }

  bks = c(-300, seq(20, 75, 5), 300)

  # df_help[1, 2:8] = table(cut(data_hist_rich$Temp, breaks = iof_breaks))

  result <- dataT %>%
            mutate(thresh_range = cut(get(col_name), breaks = bks)) %>%
            tidyr::complete(time_period, thresh_range, model, scenario, city) %>%
            group_by(time_period, thresh_range, model, scenario, city) %>%
            summarize(no_years = n_distinct(chill_season, na.rm = TRUE)) %>% 
            data.table()
  
  time_periods = c("Historical", "2025_2050", "2051_2075", "2076_2099")
  result$time_period = factor(result$time_period, levels=time_periods, order=T)
  
  result$thresh_range <- factor(result$thresh_range, order=T)
  result$thresh_range <- fct_rev(result$thresh_range)
  result <- result[order(thresh_range), ]

  result <- result %>% 
            group_by(time_period, model, scenario, city) %>% 
            mutate(n_years_passed = cumsum(no_years)) %>% 
            data.table()
  
  result_hist <- result %>% filter(time_period == "Historical") %>% data.table()
  result_50 <- result %>% filter(time_period == "2025_2050") %>% data.table()
  result_75 <- result %>% filter(time_period == "2051_2075") %>% data.table()
  result_99 <- result %>% filter(time_period == "2076_2099") %>% data.table()
  
  result_hist$frac_passed = result_hist$n_years_passed / h_year_count
  result_50$frac_passed = result_50$n_years_passed / f1_year_count
  result_75$frac_passed = result_75$n_years_passed / f2_year_count
  result_99$frac_passed = result_99$n_years_passed / f3_year_count

  result <- rbind(result_hist, result_50, result_75, result_99)
  result <- na.omit(result)
  return(result)
}

plot_boxes <- function(p_data, due, noch=FALSE, start){
  color_ord = c("grey70" , "dodgerblue", "olivedrab4", "red") # 
  time_lab = c("Historical", "2025-2050", "2051-2075", "2076-2099")
  
  box_width = 0.8
  if (due == "Jan"){
    title_s = "Thresholds met by Jan. 1st"
    } else if (due == "Feb") {
      title_s = "Thresholds met by Feb. 1st"
    } else if (due == "Mar"){
      title_s = "Thresholds met by Mar. 1st"
    } else if (due == "Apr"){
      title_s = "Thresholds met by Apr. 1st"
  }
  # reverse the order of thresholds so
  # they appear from small to large in the plot
  # We can rename them as well.
  p_data$thresh_range <- fct_rev(p_data$thresh_range)

  thresh_lab <- levels(p_data$thresh_range)
  thresh_lab <- unlist(strsplit(thresh_lab, ","))
  thresh_lab <- thresh_lab[c(TRUE, FALSE)]
  thresh_lab <- unlist(strsplit(thresh_lab, "(", fixed=T))
  thresh_lab <- thresh_lab[c(FALSE, TRUE)]

  # do the following so historical data appear in both RCP's subplots
  p_data_f <- p_data %>% filter(scenario != "Historical")
  p_data_h_45 <- p_data %>% filter(scenario == "Historical")
  p_data_h_85 <- p_data %>% filter(scenario == "Historical")
  p_data_h_45$scenario = "RCP 4.5"
  p_data_h_85$scenario = "RCP 8.5"
  p_data = rbind(p_data_h_45, p_data_h_85, p_data_f)

  the_theme <-theme_bw() + 
              theme(plot.margin = unit(c(t=.2, r=.2, b=.2, l=0.2), "cm"),
                    panel.border = element_rect(fill=NA, size=.3),
                    panel.grid.major = element_line(size = 0.05),
                    panel.grid.minor = element_blank(),
                    panel.spacing.y = unit(.35, "cm"),
                    panel.spacing.x = unit(.25, "cm"),
                    legend.position = "bottom", 
                    legend.key.size = unit(1, "line"),
                    legend.spacing.x = unit(.2, 'cm'),
                    legend.text = element_text(size=11),
                    legend.margin = margin(t=0, r=0, b=0, l=0, unit = 'cm'),
                    legend.title = element_blank(),
                    strip.text.x = element_text(size=10),
                    strip.text.y = element_text(size=10),
                    axis.ticks = element_line(size=.1, color="black"),
                    # axis.text.x = element_blank(), # element_text(size=7, face="plain", color="black"),
                    axis.text.y = element_text(size=10, face="plain", color="black"),

                    axis.title.x = element_text(size=13, face="plain", margin = margin(t=10, r=0, b=0, l=0)),
                    axis.title.y = element_text(size=13, face="plain", margin = margin(t=0, r=8, b=0, l=0))
                    )
  box <- ggplot(data = p_data, aes(x=thresh_range, y=frac_passed, fill=time_period)) +
         geom_boxplot(outlier.size = -.3, notch= noch, width=box_width, lwd=.1) +
         labs(x = "thresholds", y = "chill portion fraction met") +
         facet_grid(~ scenario ~ city) + 
         scale_fill_manual(values = color_ord,
                           name = "Time\nPeriod", 
                           labels = time_lab) + 
         scale_x_discrete(#breaks = x_breaks,
                          labels = thresh_lab)  +
         ggtitle(title_s)  +
         the_theme

  output_name <- paste0(start, "_start_", due, "_thresholds.png")
  ggsave(output_name, box, 
         path=plot_path, width=20, height=4, unit="in", dpi=400)
  return(box)
}

##################
##################
##################      Driver
##################
##################

main_in <- "/Users/hn/Desktop/Desktop/Kirti/check_point/chilling/"
setwd(main_in)
param_d <- "/Users/hn/Documents/GitHub/Kirti/Chilling/parameters/"
starts <- c("sept", "mid_sept", "oct", "mid_oct", "nov", "mid_nov")

for (st in starts){
  file = paste0(st, "_summary_comp.rds")
  mdata <- data.table(readRDS(file))
  mdata <- mdata %>% filter(model != "observed")
  
  ########################################################
  #
  # Pick up chosen cities!
  #
  ########################################################
  mdata <- pick_single_cities(mdata, param_d)

  information <- clean_process(mdata)
  jan_data = information[[1]]
  feb_data = information[[2]]
  mar_data = information[[3]]
  apr_data = information[[4]]
  rm(information, mdata)

  jan_result = count_years_threshs_met(dataT = jan_data, due="Jan")
  feb_result = count_years_threshs_met(dataT = feb_data, due="Feb")
  mar_result = count_years_threshs_met(dataT = mar_data, due="Mar")
  apr_result = count_years_threshs_met(dataT = apr_data, due="Apr")
  rm(jan_data, feb_data, mar_data)

  jan_plot <- plot_boxes(p_data=jan_result, due="Jan", noch=F, start=st)
  feb_plot <- plot_boxes(p_data=feb_result, due="Feb", noch=F, start=st)
  mar_plot <- plot_boxes(p_data=mar_result, due="Mar", noch=F, start=st)
  apr_plot <- plot_boxes(p_data=apr_result, due="Apr", noch=F, start=st)

  big_plot <- ggarrange(jan_plot, 
                        feb_plot,
                        mar_plot,
                        apr_plot, 
                        label.x = "threshold",
                        label.y = "chill portion fraction met",
                        ncol = 1, 
                        nrow = 4, 
                        common.legend = T,
                        legend = "bottom")
  ggsave(paste0(st, "_start_all_in_one.png"),
         big_plot,
         path=plot_path, 
         width=20, height=12, unit="in", dpi=300)
}








