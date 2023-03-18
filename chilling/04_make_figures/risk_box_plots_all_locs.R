
rm(list=ls())
.libPaths("/data/hydro/R_libs35")
.libPaths()

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

clean_process_all_locs <- function(dt){
  dt <- subset(dt, select=c(chill_season, sum_J1, 
                            sum_F1, sum_M1, sum_A1,lat, long, warm_cold,
                            scenario, model, year, location))
  
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
  
  jan_data <- subset(dt, select=c(sum_J1, scenario, model, time_period, chill_season, warm_cold, location))
  feb_data <- subset(dt, select=c(sum_F1, scenario, model, time_period, chill_season, warm_cold, location))
  mar_data <- subset(dt, select=c(sum_M1, scenario, model, time_period, chill_season, warm_cold, location))
  apr_data <- subset(dt, select=c(sum_A1, scenario, model, time_period, chill_season, warm_cold, location))
  return (list(jan_data, feb_data, mar_data, apr_data))
}

plot_boxes <- function(p_data, due, noch=FALSE, start){
  color_ord <- c("grey47", "dodgerblue", "olivedrab4", "red")
  time_lab = c("Historical", "2025-2050", "2051-2075", "2076-2099")
  
  box_width = .7
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
  thresh_lab[thresh_lab=="-300"] = "< 20"
  thresh_lab[thresh_lab=="300"] = "> 75"

  the_theme <-theme(plot.margin = unit(c(t=.2, r=.2, b=.2, l=0.2), "cm"),
                    panel.border = element_rect(fill=NA, size=.3),
                    panel.grid.major = element_line(size = 0.05),
                    panel.grid.minor = element_blank(),
                    panel.spacing.y = unit(.35, "cm"),
                    panel.spacing.x = unit(.25, "cm"),
                    legend.position = "bottom", 
                    legend.key.size = unit(1.2, "line"),
                    legend.spacing.x = unit(.2, 'cm'),
                    legend.text = element_text(size=14),
                    legend.margin = margin(t=0, r=0, b=0, l=0, unit = 'cm'),
                    legend.title = element_blank(),
                    plot.title = element_text(size=14, face="bold"),
                    strip.text.x = element_text(size=14, face="bold"),
                    strip.text.y = element_text(size=14, face="bold"),
                    axis.ticks = element_line(size=.1, color="black"),
                    axis.text.x = element_text(size=14, face="bold", color="black"),                   
                    axis.text.y = element_text(size=14, face="bold", color="black"),
                    axis.title.x = element_text(size=18, face="bold", margin = margin(t=10, r=0, b=0, l=0)),
                    axis.title.y = element_text(size=18, face="bold", margin = margin(t=0, r=8, b=0, l=0))
                    )
  box <- ggplot(data = p_data, aes(x=thresh_range, y=frac_passed, fill=time_period)) +
         geom_boxplot(outlier.size = -.3, notch= noch, width=box_width, lwd=.1) +
         labs(x = "thresholds", y = "chill portion fraction met") +
         facet_grid(~ scenario ~ warm_cold) + 
         scale_fill_manual(values = color_ord,name = "Time\nPeriod", labels = time_lab) + 
         scale_x_discrete(labels = thresh_lab)  +
         ggtitle(title_s)  +
         the_theme

  output_name <- paste0(start, "_start_", due, "_thresholds.png")
  ggsave(output_name, box, width=30, height=4, unit="in", dpi=400)
  return(box)
}

##################
##################
##################      Driver
##################
##################

main_in <- "/Users/hn/Desktop/Desktop/Kirti/check_point/chilling/"
# main_in <- "/data/hydro/users/Hossein/chill/data_by_core/dynamic/02/"
setwd(main_in)

param_d <- "/Users/hn/Documents/GitHub/Kirti/Chilling/parameters/"
# param_d <- "/home/hnoorazar/chilling_codes/parameters/"
starts <- c("sept", "mid_sept", "oct", "mid_oct", "nov", "mid_nov") # 

LocationGroups_NoMontana <- read.csv(paste0(param_d, "LocationGroups_NoMontana.csv"), header=T, as.is=T)

for (st in starts){
  file = paste0( st, "_summary_comp.rds") # st, "/", <- use this for Aeolus, add it to the beginning
  mdata <- data.table(readRDS(file))
  mdata <- mdata %>% filter(model != "observed")
  
  # remove Montana, add Wamr, Cool, Oregon
  mdata <- remove_montana_add_warm_cold(mdata, LocationGroups_NoMontana)
  information <- clean_process_all_locs(mdata)

  jan_data = information[[1]]
  feb_data = information[[2]]
  mar_data = information[[3]]
  apr_data = information[[4]]
  rm(information, mdata)

  jan_result = count_years_threshs_met_all_locations(dataT = jan_data, due="Jan")
  feb_result = count_years_threshs_met_all_locations(dataT = feb_data, due="Feb")
  mar_result = count_years_threshs_met_all_locations(dataT = mar_data, due="Mar")
  apr_result = count_years_threshs_met_all_locations(dataT = apr_data, due="Apr")
  rm(jan_data, feb_data, mar_data)

  warm_cold <- subset(LocationGroups_NoMontana, select=c(location, warm_cold))

  jan_result <- left_join(jan_result, warm_cold)
  feb_result <- left_join(feb_result, warm_cold)
  mar_result <- left_join(mar_result, warm_cold)
  apr_result <- left_join(apr_result, warm_cold)
  
  areas_oder <- c("Cooler Areas", "Warmer Areas", "Oregon Areas")

  jan_result$warm_cold <- factor(jan_result$warm_cold, levels= areas_oder, order=T)
  feb_result$warm_cold <- factor(feb_result$warm_cold, levels= areas_oder, order=T)
  mar_result$warm_cold <- factor(mar_result$warm_cold, levels= areas_oder, order=T)
  apr_result$warm_cold <- factor(apr_result$warm_cold, levels= areas_oder, order=T)

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
         big_plot, width=30, height=14, unit="in", dpi=300)
}

