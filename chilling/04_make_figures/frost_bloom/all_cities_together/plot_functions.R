
library(data.table)
library(dplyr)
library(tidyverse)
library(lubridate)
library(ggpubr)

options(digits=9)
options(digit=9)

##########################################################################################

plot_frost_TS <- function(dt, colname){ # This is used for plotting
  p <- ggplot(data=dt) + 
       geom_point(aes(x = year, y = get(colname)), alpha = 0.25, shape = 21, size = 1) + 
       geom_line(aes(x = year, y = get(colname))) + 
       geom_smooth(aes(x = year, y = get(colname)), method = "lm", size=1, se=F) + 
       facet_grid(~ emission ~ city ~ model) +
       theme(plot.title = element_text(size=13, face="bold"),
             plot.margin = margin(1, 1, 1, 1, "cm"),
             panel.grid.minor = element_blank(),
             legend.position = "none",
             panel.spacing = unit(.5, "cm"),
             panel.grid.major = element_line(size = 0.1),
             axis.ticks = element_line(color = "black", size = .2),
             strip.text.x = element_text(size = 28, face = "bold"),
             strip.text.y = element_text(size = 20, face = "bold"),
             axis.text.x = element_text(size = 24, face = "plain", color="black", angle=-30),
             axis.text.y = element_text(size = 24, color="black"),
             axis.title.x = element_text(size = 28, face = "bold", 
                                         margin = margin(t=12, r=0, b=0, l=0)),    
             axis.title.y = element_text(size = 28, face = "bold", 
                                         margin = margin(t=0, r=12, b=0, l=0)),
             axis.ticks.y = element_blank()) + 
       labs(y = "day count (starting Sept. 1)")
  return (p)
}

boxplot_frost_dayofyear <- function(dt, colname, kth_day, sub_title){ # This is used for plotting

  box_width <- 0.25
  color_ord = c("grey40", "dodgerblue", "olivedrab4", "darkviolet", "black", "red")
  categ_lab = c("1979-2015 (observed)", "1950-2005 (historical)", 
                "2006-2025", "2026-2050", "2051-2075", "2076-2095")

  if (kth_day==1){
    title_ <- "First frost day"
   } else if (kth_day==5){
    title_ <- "Fifth frost day"
  }

  df <- data.frame(dt)
  df <- (df %>% group_by(time_period, emission))
  medians <- (df %>% summarise(med = median(chill_dayofyear)))

  the_theme <- theme(plot.title = element_text(size=13, face="bold"),
                     panel.grid.minor = element_blank(),
                     panel.spacing=unit(.5, "cm"),
                     legend.margin=margin(t=.1, r = 0, b = .1, l = 0, unit = 'cm'),
                     legend.title = element_blank(),
                     legend.position="bottom", 
                     legend.key.size = unit(1.5, "line"),
                     legend.spacing.x = unit(.05, 'cm'),
                     panel.grid.major = element_line(size = 0.1),
                     axis.ticks = element_line(color="black", size = .2),
                     strip.text = element_text(size=12, face = "bold"),
                     legend.text=element_text(size=12),
                     axis.title.x = element_text(size= 12, face = "bold",  margin = margin(t=8, r=0, b=0, l=0)),
                     axis.title.y = element_text(size= 12, face = "bold",  margin = margin(t=0, r=8, b=0, l=0)),
                     axis.text.x = element_blank(), # element_text(size= 12, face = "plain", color="black", angle=-30),
                     axis.text.y = element_text(size=12, color="black"),
                     axis.ticks.y = element_blank())

  safe_b <- ggplot(data = dt, aes(x=time_period, y= get(colname), fill=time_period)) +
            geom_boxplot(outlier.size = -.05, notch=F, width=box_width, lwd=.3) +
            facet_grid(~ emission) +
            labs(x="", y="day count (starting Sept. 1)") +
            the_theme + 
            scale_fill_manual(values = color_ord,
                              name = "Time\nPeriod", 
                              labels = categ_lab) + 
            scale_color_manual(values = color_ord,
                               name = "Time\nPeriod", 
                               limits = color_ord,
                               labels = categ_lab) + 
            scale_x_discrete(breaks = c("1979-2016", "1950-2005", "2006-2025", 
            	                          "2026-2050", "2051-2075", "2076-2095"),
                             labels = categ_lab)  +
            ggtitle(lab=title_, subtitle=sub_title) + 
            geom_text(data = medians, 
                      aes(label = sprintf("%1.0f", medians$med), y=medians$med), 
                          colour = "white", fontface = "bold",
                          size=3,
                          position = position_dodge(.09),
                          vjust = -1.2)

  return(safe_b)
}
