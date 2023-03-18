#
# This is a copy of the same file from /Users/hn/Documents/00_GitHub/Ag/chilling/06_fill_in_paper_details
# which includes more scripts!
#
rm(list=ls())

library(data.table)
library(dplyr)
library(ggpubr)
library(ggplot2)
###########################################
post_fix <- "/0_replaced_with_367/"
            
data_dir <- "/Users/hn/Documents/01_research_data/Ag_Papers_data/Chill_Paper/tables/table_for_ugly_lines_plots/"
data_dir <- paste0(data_dir, post_fix)

param_dir <- "/Users/hn/Documents/00_GitHub/Ag/chilling/"

###########################################

data <- data.table(read.csv(file=paste0(data_dir, "medians.csv"), header=TRUE, as.is=TRUE))
DoY_map <- read.csv(paste0(param_dir, "chill_DoY_map.csv"), as.is=TRUE)

###########################################

#
# clean data
#
setnames(data, old=c("thresh_20_med", "thresh_25_med",
                     "thresh_30_med", "thresh_35_med",
                     "thresh_40_med", "thresh_45_med",
                     "thresh_50_med", "thresh_55_med",
                     "thresh_60_med", "thresh_65_med",
                     "thresh_70_med", "thresh_75_med"), 
               new=c("20", "25", "30", "35", "40", 
                     "45", "50", "55", "60", "65", "70", "75"))

data <- data %>% filter(time_period != "2006-2025") %>% data.table()
data$time_period[data$time_period == "1979-2015"] <- "Historical"

time_periods = c("Historical", "2026-2050", "2051-2075", "2076-2099")
data$time_period <- factor(data$time_period, levels = time_periods, order=TRUE)

data_melt = melt(data, id=c("city", "emission", "time_period"))

# Convert the column variable to integers
data_melt[,] <- lapply(data_melt, factor)
data_melt[,] <- lapply(data_melt, function(x) type.convert(as.character(x), as.is = TRUE))
time_periods = c("Historical", "2026-2050", "2051-2075", "2076-2099")
data_melt$time_period <- factor(data_melt$time_period, levels = time_periods, order=TRUE)

ict <- c("Omak", "Yakima", "Walla Walla", "Eugene")

data_melt <- data_melt %>% 
             filter(city %in% ict) %>% 
             data.table()

data_melt$city <- factor(data_melt$city, levels = ict, order=TRUE)


tickSize = 16
axlabelSize = 18
the_thm <- theme(plot.margin = unit(c(t=.2, r=.2, b=.2, l=0.2), "cm"),
                 panel.border = element_rect(fill=NA, size=.3),
                 panel.grid.major = element_line(size = 0.05),
                 panel.grid.minor = element_blank(),
                 panel.spacing = unit(.25, "cm"),
                 legend.position = "bottom", 
                 legend.key.size = unit(1.5, "line"),
                 legend.spacing.x = unit(.05, 'cm'),
                 panel.spacing.y = unit(.5, 'cm'),
                 legend.text = element_text(size=axlabelSize),
                 legend.margin = margin(t=0, r=0, b=0, l=0, unit = 'cm'),
                 legend.title = element_blank(),
                 plot.title = element_text(size=axlabelSize, face = "bold"),
                 plot.subtitle = element_text(face = "bold"),
                 strip.text.x = element_text(size=axlabelSize, face="bold"),
                 strip.text.y = element_text(size=axlabelSize, face="bold"),
                 axis.ticks = element_line(size=.1, color="black"),
                 axis.title.x = element_text(size = axlabelSize, face="bold", margin = margin(t=10, r=0, b=0, l=0)),
                 axis.title.y = element_text(size = axlabelSize, face="bold", margin = margin(t=0, r=10, b=0, l=0)),
                 axis.text.x = element_text(size = tickSize, face="plain", color="black", angle=90, hjust = 1),
                 axis.text.y = element_text(size = tickSize, face="plain", color="black")
                )

color_ord <- c("black", "dodgerblue", "olivedrab4", "tomato1")
color_ord <- c("grey47" , "dodgerblue", "olivedrab4", "red") #

plot_path <- "/Users/hn/Documents/00_GitHub/ag_papers/chill_paper/02_Springer_2/figure_200/"

if (dir.exists(plot_path) == F) {
  dir.create(path = plot_path, recursive = T)
}


x_start = 30
qual = 400

data_melt$emission <- factor(data_melt$emission, 
                                levels=c("RCP 8.5", "RCP 4.5"),
                                order=TRUE)

plot = ggplot(data_melt, aes(y=variable, x=value), fill=factor(time_period)) + 
       geom_path(aes(colour = factor(time_period))) + 
       facet_grid(~ emission ~ city, scales = "free") + 
       labs(y = "accumulated chill portions", x = "day of year", fill = "Climate Group") +
       scale_color_manual(labels = time_periods, values = color_ord) + 
       scale_x_continuous(breaks = DoY_map$day_count_since_sept, labels= DoY_map$letter_day) + 
       scale_y_continuous(limits = c(20, 75), breaks = seq(20, 80, by = 10)) +
       the_thm + 
       coord_cartesian(xlim = c(min(data_melt$value), max(data_melt$value))) 
plot


plot = ggplot(data_melt, aes(x=variable, y=value), fill=factor(time_period)) + 
       geom_path(aes(colour = factor(time_period))) + 
       facet_grid(~ emission ~ city, scales = "free") + 
       labs(y = "accumulated chill portions", x = "day of year", fill = "Climate Group") +
       scale_color_manual(labels = time_periods, values = color_ord) + 
       scale_y_continuous(breaks = DoY_map$day_count_since_sept, labels= DoY_map$letter_day) + 
       scale_x_continuous(limits = c(x_start, 75), breaks = seq(x_start, 80, by = 10)) +
       the_thm + 
       coord_cartesian(ylim = c(min(data_melt$value), max(data_melt$value))) 
plot


qual = 400
output_name = paste0("median_DoY_thresh_start_", x_start, ".pdf")
ggsave(filename=output_name, plot=plot, device="pdf", 
     path=plot_path, 
     width=12, height=7, unit="in", dpi=qual)






