#########################################################################################
###############################
###############################         population plots
###############################
#########################################################################################

#########################################################
#########
######### population at the beginning, end and difference
#########
#########################################################
rm(list=ls())
library(data.table)
library(ggplot2)
library(dplyr)

part_1 = "/Users/hn/Desktop/Desktop/Kirti/check_point/my_aeolus_2015/all_local/"
part_2 = "pest_control/5th_try_1000F_data/"
data_dir = paste0(part_1, part_2)
start_end_name = "start_end_1000F_"
# full_window_name = "all_14_days_window_"
models = c("45", "85")
color_ord = c("grey70", "dodgerblue", "olivedrab4", "red")
box_width = 0.3 
the_theme <- theme(plot.margin = unit(c(t=0.1, r=0.1, b=.1, l=0.1), "cm"),
                   panel.border = element_rect(fill=NA, size=.3),
                   panel.grid.major = element_line(size = 0.05),
                   panel.grid.minor = element_blank(),
                   panel.spacing=unit(.25,"cm"),
                   legend.position="bottom", 
                   legend.title = element_blank(),
                   legend.key.size = unit(.75, "line"),
                   legend.text=element_text(size=5),
                   legend.margin=margin(t= -.3, r = 0, b = 0, l = 0, unit = 'cm'),
                   legend.spacing.x = unit(.05, 'cm'),
                   strip.text.x = element_text(size = 5),
                   axis.ticks = element_line(color = "black", size = .2),
                   #axis.text = element_text(face = "plain", size = 2.5, color="black"),
                   #axis.title.x = element_text(face = "plain", size=6, margin = margin(t=2.5, r=0, b=0, l=0)),
                   axis.title.x=element_blank(),
                   axis.text.x = element_text(size = 5, face = "plain", color="black"),
                   axis.ticks.x = element_blank(),
                   axis.title.y = element_text(face = "plain", size = 8, 
                                               margin = margin(t=0, r=.1, b=0, l=0)),
                   axis.text.y = element_text(size = 5, face="plain", color="black")
                   # axis.title.y = element_blank()
                   )
                     
for (model in models){
  data = data.table(readRDS(paste0(data_dir, start_end_name, model, ".rds")))
  data$total_larva_pop_start = data$PercLarvaGen1_start + data$PercLarvaGen2_start
  data$total_larva_pop_end   = data$PercLarvaGen1_end   + data$PercLarvaGen2_end
  data$pop_delta = data$total_larva_pop_end - data$total_larva_pop_start
  data_all_scenarios <- subset(data, select=c("ClimateScenario", "ClimateGroup", 
                                              "CountyGroup", 
                                              "total_larva_pop_start", 
                                              "total_larva_pop_end",
                                              "pop_delta"))

  data <- subset(data, select=c("ClimateGroup", "CountyGroup",
                                "total_larva_pop_start", "total_larva_pop_end",
                                "pop_delta"))
  data <- melt(data, id=c("ClimateGroup", "CountyGroup"))
  y_lims <- c(0, max(data$value))

  df <- data.frame(data)
  df <- (df %>% group_by(CountyGroup, ClimateGroup, variable))
  medians <- (df %>% summarise(med = median(value)))
  options(digits=9)
  box_plot = ggplot(data = data, aes(x = variable, y = value, fill = ClimateGroup)) + 
             geom_boxplot(outlier.size= -.3, lwd=0.1, 
                          notch=TRUE, width=box_width, 
                          position=position_dodge(.8)) +
             # The bigger the number in expand below, the smaller the space between y-ticks
             labs(x="", y="Population fraction", color = "Climate Group") +
             facet_wrap(~CountyGroup) +
             scale_x_discrete(expand=c(0, .5), limits = levels(data$variable[1]), 
                              labels = c("Starting pop.", "14th days pop.", "Pop. diff.")
                              ) +
             scale_y_continuous(limits = y_lims, 
                                breaks=seq(y_lims[1], y_lims[2], by=.1)) + 
             theme_bw() +
             the_theme +
             scale_fill_manual(values=color_ord, name="Time\nperiod") + 
             scale_color_manual(values=color_ord,name="Time\nperiod", 
                                limits = color_ord)  + 
             geom_text(data = medians, 
                       aes(label = sprintf("%1.4f", medians$med), y=medians$med+.01), 
                       size=1.2, 
                       position =  position_dodge(.8),
                       vjust = 0)

  plot_path = data_dir
  plot_name = paste0("14days_1000F_pop_median_", model, ".png")
  ggsave(plot_name, box_plot, 
         path=plot_path, 
         device="png", 
         width=8, height=2, 
         units = "in", 
         dpi=400)
}
############################################################################
##
##      The same as above, except each scenatio is separated by facet!
##
##
rm(list=ls())
library(data.table)
library(ggplot2)
library(dplyr)

part_1 = "/Users/hn/Desktop/Desktop/Kirti/check_point/my_aeolus_2015/all_local/"
part_2 = "pest_control/5th_try_1000F_data/"
data_dir = paste0(part_1, part_2)
start_end_name = "start_end_1000F_"
# full_window_name = "all_14_days_window_"
models = c("45", "85")
color_ord = c("grey70", "dodgerblue", "olivedrab4", "red")
box_width = 0.3 
the_theme <- theme(plot.margin = unit(c(t=0.1, r=0.1, b=.1, l=0.1), "cm"),
                   panel.border = element_rect(fill=NA, size=.3),
                   panel.grid.major = element_line(size = 0.05),
                   panel.grid.minor = element_blank(),
                   panel.spacing=unit(.25,"cm"),
                   legend.position="bottom", 
                   legend.title = element_blank(),
                   legend.key.size = unit(.75, "line"),
                   legend.text=element_text(size=5),
                   legend.margin=margin(t= -.3, r = 0, b = 0, l = 0, unit = 'cm'),
                   legend.spacing.x = unit(.05, 'cm'),
                   strip.text.x = element_text(size = 5),
                   axis.ticks = element_line(color = "black", size = .2),
                   #axis.text = element_text(face = "plain", size = 2.5, color="black"),
                   #axis.title.x = element_text(face = "plain", size=6, margin = margin(t=2.5, r=0, b=0, l=0)),
                   axis.title.x=element_blank(),
                   axis.text.x = element_text(size = 5, face = "plain", color="black"),
                   axis.ticks.x = element_blank(),
                   axis.title.y = element_text(face = "plain", size = 8, 
                                               margin = margin(t=0, r=.1, b=0, l=0)),
                   axis.text.y = element_text(size = 5, face="plain", color="black")
                  # axis.title.y = element_blank()
                   )
                     
for (model in models){
  data = data.table(readRDS(paste0(data_dir, start_end_name, model, ".rds")))
  data$total_larva_pop_start = data$PercLarvaGen1_start + data$PercLarvaGen2_start
  data$total_larva_pop_end   = data$PercLarvaGen1_end   + data$PercLarvaGen2_end
  data$pop_delta = data$total_larva_pop_end - data$total_larva_pop_start
  data <- subset(data, select=c("ClimateScenario", "ClimateGroup", 
                                "CountyGroup", 
                                "total_larva_pop_start", 
                                "total_larva_pop_end",
                                "pop_delta"))

  data <- melt(data, id=c("ClimateScenario", "ClimateGroup", "CountyGroup"))
  y_lims <- c(0, max(data$value))

  df <- data.frame(data)
  df <- (df %>% group_by(ClimateScenario, CountyGroup, ClimateGroup, variable))
  medians <- (df %>% summarise(med = median(value)))
  options(digits=9)
  box_plot = ggplot(data = data, aes(x = variable, y = value, fill = ClimateGroup)) + 
             geom_boxplot(outlier.size= -.3, lwd=0.1, 
                          notch=TRUE, width=box_width, 
                          position=position_dodge(.8)) +
             # The bigger the number in expand below, the smaller the space between y-ticks
             labs(x="", y="Population fraction", color = "Climate Group") +
             facet_wrap(~ ClimateScenario ~CountyGroup) +
             scale_x_discrete(expand=c(0, .5), limits = levels(data$variable[1]), 
                              labels = c("Starting pop.", "14th days pop.", "Pop. diff.")
                              ) +
             scale_y_continuous(limits = y_lims, 
                                breaks=seq(y_lims[1], y_lims[2], by=.1)) + 
             theme_bw() +
             the_theme +
            scale_fill_manual(values=color_ord, name="Time\nperiod") + 
            scale_color_manual(values=color_ord,name="Time\nperiod", 
                               limits = color_ord)  + 
            geom_text(data = medians, 
                      aes(label = sprintf("%1.4f", medians$med), y=medians$med+.01), 
                      size=1.2, 
                      position =  position_dodge(.8),
                      vjust = 0)

  plot_path = data_dir
  plot_name = paste0("14days_1000F_pop_median_separate_", model, ".png")
  ggsave(plot_name, box_plot, 
         path=plot_path, 
         device="png", 
         width=8, height=8, 
         units = "in", 
         dpi=400)
}



