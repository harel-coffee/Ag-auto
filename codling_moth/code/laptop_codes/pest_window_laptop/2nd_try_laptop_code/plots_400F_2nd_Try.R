rm(list=ls())
library(data.table)
library(ggplot2)
library(dplyr)

part_1 = "/Users/hn/Desktop/Desktop/Kirti/check_point/my_aeolus_2015/all_local/"
part_2 = "pest_control/400_2nd_try_data/"
data_dir = paste0(part_1, part_2)
start_end_name = "start_end_"
full_window_name = "all_14_days_window_"
models = c("45", "85")

##############################################################
###############################
############################### GDD Cloud plots!
###############################
##############################################################
for (model in models){
  curr_data = data.table(readRDS(paste0(data_dir, full_window_name, model, ".rds")))
  curr_data = within(curr_data, remove(ID, PercLarvaGen1, year))
  curr_data <- curr_data[, .(CumDD = median(CumDDinF)), by = c("CountyGroup", 
                                                               "location", 
                                                               "ClimateScenario", 
                                                               "ClimateGroup", 
                                                               "dayofyear")]

  the_theme = theme(# panel.grid.major = element_line(size = 0.7),
                    # panel.grid.major = element_blank(),
                    panel.grid.minor = element_blank(),
                    legend.title = element_text(face = "plain", size = 12),
                    legend.text = element_text(size = 10),
                    legend.position = "bottom",
                    strip.text = element_text(size = 12, face = "plain"),
                    axis.text = element_text(face = "plain", size = 10, color="black"),
                    axis.title.x = element_text(face = "plain", size=16, margin=margin(t=10, r=0, b=0, l=0)),
                    axis.title.y = element_text(face = "plain", size=16, margin=margin(t=0, r=10, b=0, l=0))
                    )
  y_range = seq(0, 1000, 100)
  plot = ggplot(curr_data, aes(x=dayofyear, y=CumDD, fill=factor(ClimateGroup))) +
         labs(x = "Julian day", y = "Cumulative degree days (in F)", fill = "Climate group") +
         guides(fill=guide_legend(title="")) + 
         facet_grid(. ~ CountyGroup, scales="free") +
         stat_summary(geom="ribbon", fun.y=function(z) { quantile(z,0.5) }, 
                      fun.ymin=function(z) { quantile(z,0.1) }, 
                      fun.ymax=function(z) { quantile(z,0.9) }, alpha=0.4) +
         stat_summary(geom="ribbon", fun.y=function(z) { quantile(z,0.5) }, 
                      fun.ymin=function(z) { quantile(z,0.25) }, 
                      fun.ymax=function(z) { quantile(z,0.75) }, alpha=0.8) + 
         stat_summary(geom="line", fun.y=function(z) { quantile(z,0.5) })+ 
         scale_color_manual(values=c(rgb(29, 67, 111, max=255), 
                                     rgb(92, 160, 201, max=255), 
                                     rgb(211, 91, 76, max=255), 
                                     rgb(125, 7, 37, max=255)))+
         scale_fill_manual(values=c(rgb(29, 67, 111, max=255), 
                                    rgb(92, 160, 201, max=255), 
                                    rgb(211, 91, 76, max=255), 
                                    rgb(125, 7, 37, max=255)))+
         scale_x_continuous(breaks=seq(0, 370, 15)) +
         scale_y_continuous(breaks=y_range) +
         theme_bw() + the_theme
  out_name = paste0("gdd_cloud_", model, ".png" )
  output_dir = data_dir
  ggsave(out_name, plot, path=output_dir, width=7, height=7, unit="in", dpi=500)
}
##############################################################
###############################
############################### GDD box plots!
###############################
##############################################################
rm(list=ls())
library(data.table)
library(ggplot2)
library(dplyr)

data_dir = "/Users/hn/Desktop/Desktop/Kirti/check_point/my_aeolus_2015/all_local/pest_control/400_2nd_try/"
start_end_name = "start_end_"
full_window_name = "all_14_days_window_"
models = c("45", "85")
color_ord = c("grey70", "dodgerblue", "olivedrab4", "red")
y_lims <- c(0, 150)
box_width = 0.3 
for (model in models){
  curr_data = data.table(readRDS(paste0(data_dir, start_end_name, model, ".rds")))
  curr_data <- subset(curr_data, select=c("ClimateGroup", "CountyGroup", "temp_delta"))
  curr_data <- melt(curr_data, id=c("ClimateGroup", "CountyGroup"))
  df <- data.frame(curr_data)
  df <- (df %>% group_by(CountyGroup, ClimateGroup))
  medians <- (df %>% summarise(med = median(value)))
  rm(df)
  the_theme <- theme(plot.margin = unit(c(t=0.1, r=0.1, b=.1, l=0.2), "cm"),
                     panel.border = element_rect(fill=NA, size=.3),
                     panel.grid.major = element_line(size = 0.05),
                     panel.grid.minor = element_blank(),
                     panel.spacing=unit(.25,"cm"),
                     legend.position="bottom", 
                     legend.title = element_blank(),
                     legend.key.size = unit(.5, "line"),
                     legend.text=element_text(size=5),
                     legend.margin=margin(t= -.3, r = 0, b = 0, l = 0, unit = 'cm'),
                     legend.spacing.x = unit(.05, 'cm'),
                     strip.text.x = element_text(size = 5),
                     axis.ticks = element_line(color = "black", size = .2),
                     #axis.text = element_text(face = "plain", size = 2.5, color="black"),
                     #axis.title.x = element_text(face = "plain", size=6, margin = margin(t=2.5, r=0, b=0, l=0)),
                     axis.title.x = element_blank(),
                     # axis.text.x = element_text(size = 5, face = "plain", color="black"),
                     axis.text.x = element_blank(),
                     axis.ticks.x = element_blank(),
                     axis.title.y = element_text(face = "plain", size = 6, margin = margin(t=0, r=5, b=0, l=0)),
                     axis.text.y = element_text(size = 4, face="plain", color="black")
                     # axis.title.y = element_blank()
  )

  box_plot = ggplot(data = curr_data, aes(x=ClimateGroup, y=value, fill=ClimateGroup))+
             geom_boxplot(outlier.size=-.3, notch=TRUE, width=box_width, lwd=.1) +
             theme_bw() +
             scale_x_discrete(expand=c(0, 2), limits = levels(curr_data$ClimateGroup[1])) +
             scale_y_continuous(breaks = round(seq(100, 470, by = 50))) +
             labs(x="", y="Temp. range (in F for 14 days)", color = "Climate group") +
             facet_wrap(~CountyGroup) +
             the_theme +
             scale_fill_manual(values = color_ord,
                               name = "Time\nPeriod", 
                               labels = c("Historical","2040's","2060's","2080's")) + 
             scale_color_manual(values = color_ord,
                                name = "Time\nPeriod", 
                                limits = color_ord,
                                labels = c("Historical","2040's","2060's","2080's")) + 
             geom_text(data = medians, 
                       aes(label = sprintf("%1.0f", medians$med), y=medians$med), 
                       size=1.2, 
                       position =  position_dodge(.09),
                       vjust = -1.4)
  plot_name = paste0("gdd_box_", model)
  plot_path = data_dir
  ggsave(paste0(plot_name, ".png"), box_plot, 
         path = plot_path, 
         device = "png", 
         width = 4, height=3, 
         units = "in", 
         dpi=500)
}

##############################################################
###############################
##################################### population plots
###############################
##############################################################

#########################################################################################
###############################
############################### population at the beginning, end and difference
###############################
#########################################################################################
rm(list=ls())
library(data.table)
library(ggplot2)
library(dplyr)

data_dir = "/Users/hn/Desktop/Desktop/Kirti/check_point/my_aeolus_2015/all_local/pest_control/400_2nd_try/"
start_end_name = "start_end_"
full_window_name = "all_14_days_window_"
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
  data <- subset(data, select=c("ClimateGroup", "CountyGroup", 
                                "PercLarvaGen1_start", "PercLarvaGen1_end",
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
  plot_name = paste0("14_pop_median_", model, ".png")
  ggsave(plot_name, box_plot, 
         path=plot_path, 
         device="png", 
         width=8, height=3, 
         units = "in", 
         dpi=400)
}

#######################################################################
###############################
############################### population for the first three days
###############################
#######################################################################
rm(list=ls())
library(data.table)
library(ggplot2)
library(dplyr)

data_dir = "/Users/hn/Desktop/Desktop/Kirti/check_point/my_aeolus_2015/all_local/pest_control/400_2nd_try/"
plot_path = data_dir
start_end_name = "start_end_"
full_window_name = "all_14_days_window_"
models = c("45", "85")
color_ord = c("grey70", "dodgerblue", "olivedrab4", "red")
box_width = 0.3 
the_theme <- theme(plot.margin = unit(c(t=0.1, r=0.2, b=.1, l=0.1), "cm"),
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
  data = data.table(readRDS(paste0(data_dir, full_window_name, model, ".rds")))
  ID = data$ID

  # choose every other 15^th element
  # start_ID = ID[seq(1, length(ID), 14)]
  # end_ID = start_ID + 2
  data_start = data[seq(1, length(ID), 14)]
  data_end   = data[seq(3, length(ID), 14)]
  rm(data)
  data_start <- subset(data_start, select=c("ClimateScenario" , "ClimateGroup", 
                                            "CountyGroup", "year", "location",
                                            "PercLarvaGen1"))
  colnames(data_start)[colnames(data_start)=="PercLarvaGen1"] <- "pop_start"

  data_end <- subset(data_end, select=c("ClimateScenario" , "ClimateGroup", 
                                        "CountyGroup", "year", "location",
                                        "PercLarvaGen1"))

  colnames(data_end)[colnames(data_end)=="PercLarvaGen1"] <- "pop_3rd"

  data_start <- as.data.frame(data_start)
  data_end <- as.data.frame(data_end)

  data <- merge(data_start, data_end, by=c("ClimateScenario", "ClimateGroup", 
                                           "CountyGroup", "year", "location"))

  data = subset(data, select=c("ClimateGroup", "CountyGroup", 
                                "pop_start", "pop_3rd"))

  data$pop_diff = data$pop_3rd - data$pop_start
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
                                    labels = c("Starting pop.", "Third day pop.", "Pop. diff.")
                   ) +
                   scale_y_continuous(limits = y_lims, 
                                      breaks=seq(y_lims[1], y_lims[2], by=.1)) + 
                   theme_bw() +
                   the_theme +
                   scale_fill_manual(values=color_ord, name="Time\nperiod") + 
                   scale_color_manual(values=color_ord,name="Time\nperiod", 
                                      limits = color_ord) # + 
                   geom_text(data = medians, 
                             aes(label = sprintf("%1.4f", medians$med), y=medians$med + .001), 
                             size=1.2, 
                             position =  position_dodge(.8),
                             vjust = 0)
  plot_name = paste0("3_days_pop_", model, ".png")
  ggsave(plot_name, box_plot, 
         path=plot_path, 
         device="png", 
         width=8, height=3, 
         units = "in", 
         dpi=400)
}




