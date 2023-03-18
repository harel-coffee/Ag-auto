
#rm(list=ls())

library(data.table)
library(ggpubr)
library(tidyverse)
#library(plyr)
library(ggplot2)
##########################################################################
#               *********      WARNING !!!    *********                  #
#                                                                        #
#                                                                        #
#  dplyr, tidyverse (which loads dplyr) should be loaded before plyr     # 
#  If dplyr is loaded and plyr, then rename(.) would not work            #
# So, we do: plyr::rename(data, c("old_name" = "new_name"))              #
#                                                                        #
#                                                                        #
#               *********      WARNING !!!    *********                  #
##########################################################################

# color_ord = c("black", "dodgerblue", "olivedrab4", "tomato1")
model_dir_postfix = "_model_stats/"

time_period_types = c("non_overlapping") # , "overlapping"
model_types = c("dynamic") # , "utah"
main_data_dir = "/Users/hn/Desktop/Desktop/Kirti/check_point/chilling/"

the_theme = theme_bw() + 
            theme(panel.spacing = unit(1, "lines"),
                  panel.grid.major = element_blank(),
                  legend.text = element_text(size=9),
                  legend.title = element_blank(),
                  legend.position = "bottom",
                  legend.box = "horizontal", 
                  legend.key.size = unit(.3, "cm"),
                  legend.margin=margin(t=-.1, r=0, b=.1, l=0, unit = 'cm'),
                  plot.margin = margin(t=0.1, r=.2, b=0.1, l=.2, "cm"),
                  legend.spacing.x = unit(.1, 'cm'),
                  strip.text = element_text(size=8, face="plain"),
                  plot.title = element_text(hjust = 0.5),
                  plot.subtitle = element_text(hjust = 0.5),
                  axis.text.y = element_text(size = 8, face = "plain", color="black"),
                  axis.title.x = element_text(face = "plain", size=12, 
                                              margin = margin(t=8, r=0, b=0, l=0)),
                  axis.text.x = element_text(size = 8, face = "plain", color="black"),
                  axis.title.y = element_text(face = "plain", size = 12, 
                                              margin = margin(t=0, r=8, b=0, l=0)))

for (model_type in model_types){
  for (time_period_type in time_period_types){
    print (model_type)
    print (time_period_type)

    if (time_period_type == "non_overlapping"){
      legend_labs = c("Historical", "2025-2050", "2051-2075", "2076-2100")
      time_periods = c("Historical", "2025_2050", "2051_2075", "2076_2100")
    } else {
      legend_labs = c("Historical", "2040's", "2060's", "2080's")
      time_periods = c("Historical", "2040's", "2060's", "2080's")
    }
    
    data_dir = paste0(main_data_dir, time_period_type, "/", model_type, model_dir_postfix, "tables/")
    data <- data.table(read.csv(file=paste0(data_dir, model_type, "_medians.csv"), header=TRUE))

    # data <- plyr::rename(data, c(thresh_20_med="20", thresh_25_med="25",
    #                              thresh_30_med="30", thresh_35_med="35",
    #                              thresh_40_med="40", thresh_45_med="45",
    #                              thresh_50_med="50", thresh_55_med="55",
    #                              thresh_60_med="60", thresh_65_med="65",
    #                              thresh_70_med="70", thresh_75_med="75"
    #                               ))
    setnames(data, old=c("thresh_20_med", "thresh_25_med",
                         "thresh_30_med", "thresh_35_med",
                         "thresh_40_med", "thresh_45_med",
                         "thresh_50_med", "thresh_55_med",
                         "thresh_60_med", "thresh_65_med",
                         "thresh_70_med", "thresh_75_med"), 
                   new=c("20", "25", "30", "35", "40", "45", "50", 
                         "55", "60", "65", "70", "75"))
  
    data$scen_clim = paste0(data$scenario, "_(", data$ClimateGroup, ")")
    data = within(data, remove(scenario, ClimateGroup))    
    
    data_melt = melt(data, id=c("climate_type", "scen_clim"))
    data_melt[,] <- lapply(data_melt, factor)
    data_melt[,] <- lapply(data_melt, 
                           function(x) type.convert(as.character(x), 
                           as.is = TRUE))

    data_melt[, 3] <- lapply(data_melt[, 3], as.numeric)

    data_melt$scen_clim[data_melt$scen_clim == "historical_(Historical)"] = "Modeled Historical"
    
    data_melt$scen_clim[data_melt$scen_clim == "rcp45_(2025_2050)"] = "(RCP 4.5: 2025-2050)"
    data_melt$scen_clim[data_melt$scen_clim == "rcp45_(2051_2075)"] = "(RCP 4.5: 2051-2075)"
    data_melt$scen_clim[data_melt$scen_clim == "rcp45_(2076_2100)"] = "(RCP 4.5: 2076-2099)"
    
    data_melt$scen_clim[data_melt$scen_clim == "rcp85_(2025_2050)"] = "(RCP 8.5: 2025-2050)"
    data_melt$scen_clim[data_melt$scen_clim == "rcp85_(2051_2075)"] = "(RCP 8.5: 2051-2075)"
    data_melt$scen_clim[data_melt$scen_clim == "rcp85_(2076_2100)"] = "(RCP 8.5: 2076-2099)"

    orders = c("Modeled Historical", 
               "(RCP 4.5: 2025-2050)", "(RCP 4.5: 2051-2075)", "(RCP 4.5: 2076-2099)", 
               "(RCP 8.5: 2025-2050)", "(RCP 8.5: 2051-2075)", "(RCP 8.5: 2076-2099)")
    
    data_melt$scen_clim <- factor(data_melt$scen_clim, levels=orders, order=T)

    plot = ggplot(data_melt, aes(x=variable, y=value), fill=factor(scen_clim)) + 
           geom_line(aes(color = factor(scen_clim), size=factor(scen_clim), linetype=factor(scen_clim))) + 
           facet_grid( ~ climate_type, scales = "free") + 
           labs(x = "thresholds", y = "median DoY") +
           scale_x_continuous(limits = c(20, 75), breaks = seq(20, 80, by = 10)) +
           scale_y_continuous(limits = c(60, 200), breaks = seq(60, 200, by = 10)) +
           scale_color_manual(values=c("black", 
                                       "blue", "brown", "green4", 
                                       "red", "magenta", "orange")) + 
           # scale_size_manual(values=c(.5, .3, .3, .3, .3, .3, .3)) +
           scale_size_manual(values=c(.7, .5, .5, .5, .5, .5, .5)) +
           scale_linetype_manual(values=c("solid", 
                                          "dotdash", "dotdash", "dotdash", 
                                          "solid", "solid", "solid")) + 
           guides(col = guide_legend(ncol = 4))+
           the_theme 


    output_name = paste0(model_type, "_", time_period_type, "_", "medianDoY_thresh_nonFacet.png")
    plot_path = "/Users/hn/Desktop/"
    ggsave(filename = output_name, plot = plot, device="png", 
           path=plot_path, width=8, height=4.5, unit="in")

  }
}

###########################################
# Plot Quantile 90th

for (model_type in model_types){
  for (time_period_type in time_period_types){
    print (model_type)
    print (time_period_type)

    if (time_period_type == "non_overlapping"){
      legend_labs = c("Historical", "2025-2050", "2051-2075", "2076-2100")
      time_periods= c("Historical", "2025_2050", "2051_2075", "2076_2100")
    } else {
      legend_labs = c("Historical", "2040's", "2060's", "2080's")
      time_periods= c("Historical", "2040's", "2060's", "2080's")
    }
    
    data_dir = paste0(main_data_dir, time_period_type, "/", model_type, model_dir_postfix, "tables/")
    data <- data.table(read.csv(file=paste0(data_dir, model_type, "_quan_90.csv"), header=TRUE))

    # data <- plyr::rename(data, c(thresh_20_Q90="20", thresh_25_Q90="25",
    #                              thresh_30_Q90="30", thresh_35_Q90="35",
    #                              thresh_40_Q90="40", thresh_45_Q90="45",
    #                              thresh_50_Q90="50", thresh_55_Q90="55",
    #                              thresh_60_Q90="60", thresh_65_Q90="65",
    #                              thresh_70_Q90="70", thresh_75_Q90="75"
    #                               ))
    setnames(data, old=c("thresh_20_Q90", "thresh_25_Q90",
                         "thresh_30_Q90", "thresh_35_Q90",
                         "thresh_40_Q90", "thresh_45_Q90",
                         "thresh_50_Q90", "thresh_55_Q90",
                         "thresh_60_Q90", "thresh_65_Q90",
                         "thresh_70_Q90", "thresh_75_Q90"), 
                   new=c("20", "25", "30", "35", "40", "45", "50", 
                         "55", "60", "65", "70", "75"))

    data$scen_clim = paste0(data$scenario, "_(", data$ClimateGroup, ")")
    data = within(data, remove(scenario, ClimateGroup))
    
    data_melt = melt(data, id=c("climate_type", "scen_clim"))
    data_melt[,] <- lapply(data_melt, factor)
    data_melt[,] <- lapply(data_melt, 
                           function(x) type.convert(as.character(x), 
                           as.is = TRUE))

    data_melt[, 3] <- lapply(data_melt[, 3], as.numeric)

    data_melt$scen_clim[data_melt$scen_clim == "historical_(Historical)"] = "Modeled Historical"
    
    data_melt$scen_clim[data_melt$scen_clim == "rcp45_(2025_2050)"] = "(RCP 4.5: 2025-2050)"
    data_melt$scen_clim[data_melt$scen_clim == "rcp45_(2051_2075)"] = "(RCP 4.5: 2051-2075)"
    data_melt$scen_clim[data_melt$scen_clim == "rcp45_(2076_2100)"] = "(RCP 4.5: 2076-2099)"
    
    data_melt$scen_clim[data_melt$scen_clim == "rcp85_(2025_2050)"] = "(RCP 8.5: 2025-2050)"
    data_melt$scen_clim[data_melt$scen_clim == "rcp85_(2051_2075)"] = "(RCP 8.5: 2051-2075)"
    data_melt$scen_clim[data_melt$scen_clim == "rcp85_(2076_2100)"] = "(RCP 8.5: 2076-2099)"

    orders = c("Modeled Historical", 
               "(RCP 4.5: 2025-2050)", "(RCP 4.5: 2051-2075)", "(RCP 4.5: 2076-2099)", 
               "(RCP 8.5: 2025-2050)", "(RCP 8.5: 2051-2075)", "(RCP 8.5: 2076-2099)")
    
    data_melt$scen_clim <- factor(data_melt$scen_clim, levels=orders, order=T)

    plot = ggplot(data_melt, aes(x=variable, y=value), fill=factor(scen_clim)) + 
           geom_line(aes(color = factor(scen_clim), size=factor(scen_clim), linetype=factor(scen_clim) ) ) + 
           facet_grid( ~ climate_type, scales = "free") + 
           labs(x = "thresholds", y = "90th quantile DoY") +
           scale_x_continuous(limits = c(20, 75), breaks = seq(20, 80, by = 10)) +
           scale_y_continuous(limits = c(60, 200), breaks = seq(60, 200, by = 10)) +
           scale_color_manual(values=c("black", 
                                       "blue", "brown", "green4", 
                                       "red", "magenta", "orange")) + 
           # scale_size_manual(values=c(.5, .3, .3, .3, .3, .3, .3)) +
           scale_size_manual(values=c(.7, .5, .5, .5, .5, .5, .5)) +
           scale_linetype_manual(values=c("solid", 
                                          "dotdash", "dotdash", "dotdash", 
                                          "solid", "solid", "solid")) + 
           guides(col = guide_legend(ncol = 4))+
           the_theme 


    output_name = paste0(model_type, "_", time_period_type, "_", "90th_quan_DoY_thresh_nonFacet.png")
    plot_path = "/Users/hn/Desktop/"
    ggsave(filename = output_name, plot = plot, device="png", 
           path=plot_path, width=8, height=4.5, unit="in")

  }
}

