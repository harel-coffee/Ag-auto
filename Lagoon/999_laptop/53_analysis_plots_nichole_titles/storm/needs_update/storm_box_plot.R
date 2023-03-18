rm(list=ls())
library(lubridate)
library(ggpubr)
library(purrr)
library(tidyverse)
library(data.table)
library(dplyr)
library(ggplot2)

source_path_1 = "/Users/hn/Documents/GitHub/Ag/Lagoon/core_lagoon.R"
source_path_2 = "/Users/hn/Documents/GitHub/Ag/Lagoon/core_plot_lagoon.R"
source(source_path_1)
source(source_path_2)

options(digit=9)
options(digits=9)
########################################################################
########################################################################
in_dir <- "/Users/hn/Desktop/Desktop/Ag/check_point/lagoon/storm/"
plot_dir <- paste0(in_dir, "plots/")
           
all_storms <- readRDS(paste0(in_dir, "all_storms.rds"))
all_storms <- all_storms %>%
              filter(return_period != "1979-2016" & 
                     return_period != "2006-2025")%>%
              data.table()

all_storms <- convert_5_numeric_clusts_to_alphabet(data_tb = all_storms)
head(all_storms, 2)

box_p <- storm_box_plot(data_tb = all_storms) + 
         coord_cartesian(ylim = c(1, 16))
box_p

ggsave(filename = paste0("storm_box.png"), 
       plot = box_p, 
       width = 8, height = 3, units = "in", 
       dpi=600, device = "png",
       path=plot_dir)

#####################################################################
dt_25 <- within(all_storms, remove(five_years, ten_years, 
                                   fifteen_years, twenty_years,
                                   model, location))

categ_lab <- sort(unique(dt_25$return_period))
color_ord = c("grey47", "olivedrab4", "steelblue1", "gold")

medians <- data.frame(dt_25) %>% 
           group_by(return_period, emission, cluster) %>% 
           summarise(med_25 = median(twenty_five_years)) %>% 
           data.table()

melted <- melt(dt_25, id = c("cluster", "return_period", "emission"))

ax_txt_size <- 10; ax_ttl_size <- 12; box_width = 0.65
the <- theme(plot.margin = unit(c(t=.1, r=.2, b=.1, l=0.2), "cm"),
             panel.border = element_rect(fill=NA, size=.3),
             panel.grid.major = element_line(size = 0.05),
             panel.grid.minor = element_blank(),
             panel.spacing = unit(.35, "line"),
             legend.position = "bottom", 
             legend.key.size = unit(1.5, "line"),
             legend.spacing.x = unit(.1, 'line'),
             panel.spacing.y = unit(.5, 'line'),
             legend.text = element_text(size = ax_ttl_size),
             legend.margin = margin(t=.1, r=0, b=0, l=0, unit = 'line'),
             legend.title = element_blank(),
             plot.title = element_text(size = ax_ttl_size, face = "bold"),
             plot.subtitle = element_text(face = "bold"),
             strip.text.x = element_text(size = ax_ttl_size, face = "bold",
                                         margin = margin(.15, 0, .15, 0, "line")),
             axis.ticks = element_line(size = .1, color = "black"),
             axis.text.y = element_text(size = ax_txt_size, 
                                        face = "bold", color = "black"),
             axis.text.x = element_text(size = ax_txt_size, 
                                        face = "bold", color="black",
                                        margin=margin(t=.05, r=5, l=5, b=0,"pt")
                                        ),
             axis.title.y = element_text(size = ax_ttl_size, 
                                         face = "bold", 
                                         margin = margin(t=0, r=2, b=0, l=0)),
             axis.title.x = element_blank()
            )

box_p <- ggplot(data = melted, 
                aes(x=cluster, y=value, fill=return_period)) +
         geom_boxplot(outlier.size = -0.3, notch=F, 
                      width = box_width, lwd=.1, 
                      position = position_dodge(0.85)) +
         facet_grid(~ emission) +
         xlab("precip. group") + 
         ylab("25 yr, 24 hr design storm intensity (mm/hr)") + 
         scale_fill_manual(values = color_ord,
                           name = "Return\nPeriod", 
                           labels = categ_lab) + 
         scale_y_continuous(breaks = seq(0, 20, by=5)) + 
         the +
         geom_text(data = medians, 
                   aes(label = sprintf("%1.1f", medians$med_25), 
                        y = medians$med_25),
                   size = 2.2, vjust = -.6, fontface="bold",
                   position = position_dodge(.85))
       
ggsave(filename = paste0("storm_box_25_1.png"), 
       plot = box_p, 
       width = 8.5, height = 4.5, units = "in", 
       dpi=600, device = "png",
       path=plot_dir)

############### Same as above - No Median
ax_txt_size <- 10; ax_ttl_size <- 12; box_width = 0.6
the <- theme(plot.margin = unit(c(t=.1, r=.2, b=.1, l=0.2), "cm"),
             panel.border = element_rect(fill=NA, size=.3),
             panel.grid.major = element_line(size = 0.05),
             panel.grid.minor = element_blank(),
             panel.spacing = unit(.35, "line"),
             legend.position = "bottom", 
             legend.key.size = unit(1.5, "line"),
             legend.spacing.x = unit(.1, 'line'),
             panel.spacing.y = unit(.5, 'line'),
             legend.text = element_text(size = ax_ttl_size),
             legend.margin = margin(t=.1, r=0, b=0, l=0, unit = 'line'),
             legend.title = element_blank(),
             plot.title = element_text(size = ax_ttl_size, face = "bold"),
             plot.subtitle = element_text(face = "bold"),
             strip.text.x = element_text(size = ax_ttl_size, face = "bold",
                                         margin = margin(.15, 0, .15, 0, "line")),
             axis.ticks = element_line(size = .1, color = "black"),
             axis.text.y = element_text(size = ax_txt_size, 
                                        face = "bold", color = "black"),
             axis.text.x = element_text(size = ax_txt_size, 
                                        face = "bold", color="black",
                                        margin=margin(t=.05, r=5, l=5, b=0,"pt")
                                        ),
             axis.title.y = element_text(size = ax_ttl_size, 
                                         face = "bold", 
                                         margin = margin(t=0, r=2, b=0, l=0)),
             axis.title.x =element_blank()
            )
box_p <- ggplot(data = melted, 
                aes(x=cluster, y=value, fill=return_period)) +
         geom_boxplot(outlier.size = -0.3, notch=F, 
                      width = box_width, lwd=.1, 
                      position = position_dodge(0.85)) +
         facet_grid(~ emission) +
         xlab("precip. group") + 
         ylab("25 yr, 24 hr design storm intensity (mm/hr)") + 
         scale_fill_manual(values = color_ord,
                           name = "Return\nPeriod", 
                           labels = categ_lab) + 
         scale_y_continuous(breaks = 1:20) + 
         the
       
ggsave(filename = paste0("storm_box_25_no_ann.png"), 
       plot = box_p, 
       width = 8.5, height = 4.5, units = "in", 
       dpi=600, device = "png",
       path=plot_dir)
