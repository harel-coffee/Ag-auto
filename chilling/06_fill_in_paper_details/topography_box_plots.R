rm(list=ls())

library(ggmap)
library(ggpubr)
library(lubridate)
library(purrr)
library(scales)
library(tidyverse)
library(maps)
library(data.table)
library(dplyr)
library(sp)
options(digits=9)
options(digit=9)


A <- read.csv("/Users/hn/Documents/00_GitHub/Ag_papers/Chill_Paper/MajorRevision/annual_avgs.csv", as.is=TRUE)
A <- data.table(A)

A_precip <- within(A, remove(annual_tmean, annual_winterYR_tmean))
A_temp <- within(A, remove(annual_precip_mean, annual_winterYR_precip_mean))


setnames(A_precip, old=c("annual_precip_mean", 
                         "annual_winterYR_precip_mean"), 
                   new=c("annual avg. precip.",
                         "winter avg. precip.")
        )

setnames(A_temp, old=c("annual_tmean", 
                        "annual_winterYR_tmean"), 
                 new=c("annual avg. temp.",
                       "winter avg. temp.")
        )

A_precip_melt <- melt(A_precip, id = c("location"))
A_temp_melt <- melt(A_temp, id = c("location"))

plot_bloom_box <- function(dt, colname="value"){
  color_ord = c("grey40", "red") # 
  categ_lab = unique(dt$variable)

  df <- data.frame(dt)
  medianss <- (df %>% group_by (variable) %>% summarise(med = median(get(colname))))

  box_width <- 0.35
  the_theme <- theme(plot.title = element_text(size=13, face="bold"),
                     panel.grid.minor = element_blank(),
                     panel.spacing=unit(.5, "cm"),
                     legend.margin=margin(t=.1, r=0, b=.1, l=0, unit='cm'),
                     legend.title = element_blank(),
                     legend.position="bottom", 
                     legend.key.size = unit(1.5, "line"),
                     legend.spacing.x = unit(.05, 'cm'),
                     panel.grid.major = element_line(size = 0.1),
                     axis.ticks = element_line(color="black", size = .2),
                     strip.text = element_text(size=12, face = "bold"),
                     legend.text=element_text(size=12),
                     axis.title.x = element_blank(),
                     axis.title.y = element_blank(),
                     axis.text.x = element_blank(), # element_text(size= 12, face = "plain", color="black", angle=-30),
                     axis.text.y = element_text(size=12, color="black"),
                     axis.ticks.x = element_blank())

  ggplot(data = dt, aes(x = variable, 
                        y = get(colname), 
                        fill = variable)) +
  geom_boxplot(outlier.size = -10, 
               notch=F, width=box_width, 
               lwd=.3, alpha=.8) +
  the_theme + 
  scale_fill_manual(values = color_ord, name = "Time\nPeriod", 
                    labels = categ_lab) + 

  scale_color_manual(values = color_ord, labels = categ_lab,
                     name = "Time\nPeriod", limits = color_ord) + 
  
  # ggtitle(lab=title_) +

  geom_text(data = medianss, 
            aes(label = sprintf("%1.0f", med), 
                y=med), 
                colour = "black", fontface = "bold",
                size=5, 
                position=position_dodge(.09),
                vjust = -0.5)
}

precip_plot <- plot_bloom_box(A_precip_melt)
temp_plot <- plot_bloom_box(A_temp_melt)

temp_plot
precip_plot

temp_plot <- temp_plot + coord_cartesian(ylim = c(-1, 13))
precip_plot <- precip_plot + coord_cartesian(ylim = c(0, 2500))

plot_dir <- paste0("/Users/hn/Documents/00_GitHub/Ag_papers/Chill_Paper/MajorRevision/figures_4_revision/topography")

ggsave(filename = "temp_plot_HiRes.png",
       plot = temp_plot, width=5.5, height=5, units = "in", 
       dpi=600, device = "png", path = plot_dir)

ggsave(filename = "precip_plot_HiRes.png",
       plot = precip_plot, width=5.5, height=5, units = "in", 
       dpi=600, device = "png", path = plot_dir)



ggsave(filename = "temp_plot_LowRes.png",
       plot = temp_plot, width=5.5, height=5, units = "in", 
       dpi=300, device = "png", path = plot_dir)

ggsave(filename = "precip_plot_LowRes.png",
       plot = precip_plot, width=5.5, height=5, units = "in", 
       dpi=300, device = "png", path = plot_dir)


