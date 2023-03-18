
rm(list=ls())

library(data.table)
library(ggpubr)
library(plyr)
library(ggplot2)
###########################################
model_dir_postfix = "_model_stats/"

time_period_types = c("non_overlapping") # , "overlapping"
model_types = c("dynamic", "utah")
main_data_dir = "/Users/hn/Desktop/Desktop/Kirti/check_point/chilling/"

the_theme = theme_bw() + 
            theme(panel.spacing = unit(1, "lines"),
                  panel.grid.major = element_blank(),
                  legend.text = element_text(size=10),
                  legend.title = element_blank(),
                  legend.position = "bottom",
                  legend.key.size = unit(.3, "cm"),
                  legend.margin=margin(t=-.1, r=0, b=.1, l=0, unit = 'cm'),
                  legend.spacing.x = unit(.1, 'cm'),
                  strip.text = element_text(size=10, face="plain"),
                  plot.title = element_text(hjust = 0.5),
                  plot.subtitle = element_text(hjust = 0.5),
                  axis.text.y = element_text(size =10, face = "plain", color="black"),
                  axis.title.x = element_text(face = "plain", size=10, 
                                              margin = margin(t=8, r=0, b=0, l=0)),
                  axis.text.x = element_text(size =10, face = "plain", color="black"),
                  axis.title.y = element_text(face = "plain", size = 12, 
                                              margin = margin(t=0, r=8, b=0, l=0)))

color_ord = c("black", "dodgerblue", "olivedrab4", "tomato1")

for (model_type in model_types){
  for (time_period_type in time_period_types){
    print(model_type)
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

    data <- rename(data, c(thresh_20_med="20", thresh_25_med="25",
                           thresh_30_med="30", thresh_35_med="35",
                           thresh_40_med="40", thresh_45_med="45",
                           thresh_50_med="50", thresh_55_med="55",
                           thresh_60_med="60", thresh_65_med="65",
                           thresh_70_med="70", thresh_75_med="75"
                            ))

    data_melt = melt(data, id=c("climate_type", "scenario", "ClimateGroup"))

    # Convert the column variable to integers
    data_melt[,] <- lapply(data_melt, factor)
    data_melt[,] <- lapply(data_melt, function(x) type.convert(as.character(x), as.is = TRUE))

    data_melt$ClimateGroup = factor(data_melt$ClimateGroup)
    data_melt$ClimateGroup <- ordered(data_melt$ClimateGroup, levels = time_periods)
    
    plot = ggplot(data_melt, aes(x=variable, y=value), fill=factor(ClimateGroup)) + 
           geom_path(aes(colour = factor(ClimateGroup))) + 
           facet_grid( ~ scenario ~ climate_type, scales = "free") + 
           labs(x = "thresholds", y = "median DoY", fill = "Climate Group") +
           scale_color_manual(labels = legend_labs, values = color_ord) + 
           scale_x_continuous(limits = c(20, 75), breaks = seq(20, 80, by = 10)) +
           the_theme

    output_name = paste0(model_type, "_", time_period_type, "_", "medianDoY_thresh.png")
    plot_path = data_dir
    plot_path = "/Users/hn/Desktop/Desktop/"
    ggsave(filename=output_name, plot=plot, device="png", path=plot_path, width=5, height=5, unit="in")

  }
}
