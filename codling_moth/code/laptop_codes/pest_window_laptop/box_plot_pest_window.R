rm(list=ls())
library(data.table)
library(ggplot2)
library( dplyr)

data_25_dir <- "/Users/hn/Desktop/Kirti/check_point/my_aeolus_2015/all_local/pest_control/25/"
data_50_dir <- "/Users/hn/Desktop/Kirti/check_point/my_aeolus_2015/all_local/pest_control/50/"
data_75_dir <- "/Users/hn/Desktop/Kirti/check_point/my_aeolus_2015/all_local/pest_control/75/"

directories = c(data_25_dir, data_50_dir, data_75_dir)
# directories = directories[1]
# directories = c(data_50_dir)
color_ord = c("grey70", "dodgerblue", "olivedrab4", "red")
for (dir in directories){
  file_list = list.files(path = dir, 
                         pattern = ".rds", 
                         all.files = FALSE, 
                         full.names = FALSE, 
                         recursive = FALSE)
  y_lims <- c(0, 150)
  box_width = 0.3
  if (last(unlist(strsplit(data_75_dir, "/"))) == "75"){y_lims <- c(0, 160)}
  for (file_name in file_list){
    data = data.table(readRDS(paste0(dir, file_name)))
    data <- data[data$window_length !=0, ] 
    data <- data[data$generations== "window_gen_1" | data$generations== "window_gen_2"]
    data$generations <- factor(data$generations)
    
    df <- data.frame(data)
    df <- (df %>% group_by(CountyGroup, ClimateGroup, generations))
    medians <- (df %>% summarise(med = median(window_length)))

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
                       axis.title.y = element_text(face = "plain", size = 8, margin = margin(t=0, r=.1, b=0, l=0)),
                       axis.text.y = element_text(size = 5, face="plain", color="black")
                       # axis.title.y = element_blank()
                       )

    box_plot = ggplot(data = data, aes(x = generations, y = window_length, fill = ClimateGroup)) + 
               geom_boxplot(outlier.size= -.3, lwd=0.1, 
                            notch=TRUE, width=box_width, 
                            position=position_dodge(.8)) +
               # The bigger the number in expand below, the smaller the space between y-ticks
               labs(x="", y="Time window", color = "Climate Group") +
               facet_wrap(~CountyGroup) +
               scale_x_discrete(expand=c(0, .3), limits = levels(data$generations[1]), 
                                # labels = c("Gen. 1", "Gen. 2", "Gen. 3", "Gen. 4")
                                labels = c("Gen. 1", "Gen. 2")
               ) +
               scale_y_continuous(limits = y_lims, breaks=seq(y_lims[1], y_lims[2], by=50)) + 
               theme_bw() +
               the_theme +
               scale_fill_manual(values=color_ord, name="Time\nperiod") + 
               scale_color_manual(values=color_ord,name="Time\nperiod", limits = color_ord) + 
               geom_text(data = medians, 
                         aes(label = sprintf("%1.0f", medians$med), y=medians$med+.5), 
                         size=1.2, 
                         position =  position_dodge(.8),
                         vjust = 0)

    plot_name = unlist(strsplit(file_name, ".rds"))[1]
    plot_path = dir
    ggsave(paste0(plot_name, ".png"), box_plot, 
           path=plot_path, 
           device="png", 
           width=4, height=3, 
           units = "in", 
           dpi=400)
  }
}



