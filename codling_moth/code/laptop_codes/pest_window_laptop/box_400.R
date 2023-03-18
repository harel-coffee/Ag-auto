rm(list=ls())
library(data.table)
library(ggplot2)
library(dplyr)

data_dir <- "/Users/hn/Desktop/Kirti/check_point/my_aeolus_2015/all_local/pest_control/"

file_list = c("window_400F_sub_melt_45.rds", "window_400F_sub_melt_85.rds")
color_ord = c("grey70", "dodgerblue", "olivedrab4", "red")
y_lims <- c(0, 150)
box_width = 0.3 

for (file in file_list){
  data <- data.table(readRDS(paste0(data_dir, file)))
  data <- data[data$variable=="temp_delta", ]
  df <- data.frame(data)
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

  box_plot = ggplot(data = data, aes(x=ClimateGroup, y=value, fill=ClimateGroup))+
             geom_boxplot(outlier.size=-.3, notch=TRUE, width=box_width, lwd=.1) +
             theme_bw() +
             scale_x_discrete(expand=c(0, 2), limits = levels(data$ClimateGroup[1])) +
             scale_y_continuous(breaks = round(seq(100, 470, by = 50))) +
             labs(x="", y="Temp. range (in F for 14 days)", color = "Climate group") +
             facet_wrap(~CountyGroup) +
             the_theme +
             scale_fill_manual(values=color_ord,
                               name="Time\nPeriod", 
                               labels=c("Historical","2040's","2060's","2080's")) + 
             scale_color_manual(values=color_ord,
                                name="Time\nPeriod", 
                                limits = color_ord,
                                labels=c("Historical","2040's","2060's","2080's")) + 
             geom_text(data = medians, 
                       aes(label = sprintf("%1.0f", medians$med), y=medians$med), 
                           size=1.2, 
                           position =  position_dodge(.09),
                           vjust = -1.4)

  a = unlist(strsplit(file, ".rds"))[1]
  plot_name = paste0("box_rcp", substr(a, nchar(a)-1, nchar(a)))
  plot_path = data_dir
  ggsave(paste0(plot_name, ".png"), box_plot, 
         path=plot_path, 
         device="png", 
         width=4, height=3, 
         units = "in", 
         dpi=500)

}



