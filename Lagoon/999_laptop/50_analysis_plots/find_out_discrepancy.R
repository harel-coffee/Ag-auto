rm(list=ls())
library(lubridate)
library(ggpubr)
library(purrr)
library(tidyverse)
library(data.table)
library(dplyr)
library(ggplot2)
options(digit=9)
options(digits=9)
##################################################
#
#
# discrepancy between diff of medians and the map


##################################################


in_dir <- "/Users/hn/Desktop/Desktop/Kirti/check_point/lagoon/runoff/plots/"
param_dir <- "/Users/hn/Documents/GitHub/Kirti/Lagoon/parameters/"
file <- "chunky_median_diffs_for_map.csv"

median_diffs <- read.csv(paste0(in_dir, file), header=T, as.is=T)
clusters <- read.csv(paste0(param_dir, "observed_clusters.csv"), header=T, as.is=T)
clusters <- within(clusters, remove(ann_prec_mean, centroid))

chunk_median_diffs <- merge(median_diffs, clusters, by="location", all.x=T)
chunk_median_diffs <- cluster_numeric_2_str(chunk_median_diffs)

least_precip_diff <- chunk_median_diffs %>% 
                     filter(cluster=="least precip")%>%
                     data.table()

least_precip_diff_2099 <- least_precip_diff %>% 
                          filter(time_period =="2076-2099") %>%
                          data.table()

least_precip_diff_2099_85 <- least_precip_diff_2099 %>% 
                             filter(emission =="RCP 8.5") %>%
                             data.table()



dt <- chunk_median_diffs
tgt_col <- "perc_diff"

dt_medians <- data.frame(dt) %>% 
               group_by(cluster, time_period, emission) %>% 
               summarise( med = median(get(tgt_col))) %>% 
               data.table()

dt <- within(dt, remove(diff))

melted <- melt(dt, id = c("location", 
                          "time_period", "emission",
                          "cluster"))

time_label <- c("2026-2050", "2051-2075", "2076-2099")
color_ord = c("red", "grey47", "dodgerblue2")

categ_label <- c("most precip", "less precip", 
                 "lesser precip", "least precip")

melted$cluster <- factor(melted$cluster, levels=categ_label)
melted$time_period <- factor(melted$time_period, levels=time_label)

ax_txt_size <- 8; ax_ttl_size <- 10; box_width = 0.6

the <- theme(plot.margin = unit(c(t=.1, r=.2, b=.1, l=0.2), "cm"),
             panel.border = element_rect(fill=NA, size=.3),
             panel.grid.major = element_line(size = 0.05),
             panel.grid.minor = element_blank(),
             panel.spacing = unit(.35, "line"),
             legend.position = "bottom", 
             legend.key.size = unit(.8, "line"),
             legend.spacing.x = unit(.1, 'line'),
             panel.spacing.y = unit(.5, 'line'),
             legend.text = element_text(size = ax_ttl_size, face="bold"),
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
########
diff_box <- ggplot(data = melted, aes(x=cluster, y=value, fill=time_period)) +
            the + 
            geom_boxplot(outlier.size = - 0.3, notch=F, 
                         width = box_width, lwd=.1, 
                         position = position_dodge(0.8)) +
            scale_x_discrete(expand=c(0.1, 0)) + 
            # labs(x="", y="") + # theme_bw() + 
            facet_grid(~ emission) +
            xlab("precip. group") +
            ylab("diff of percentages") + 
            scale_fill_manual(values = color_ord, labels = time_label) +
            geom_text(data = dt_medians, 
                      aes(label = sprintf("%1.0f", dt_medians$med), y = dt_medians$med), 
                      size = 2, fontface = "bold",
                      position = position_dodge(.8), vjust = -.6)


plot_dir = "/Users/hn/Desktop/Desktop/Kirti/check_point/lagoon/runoff/plots/"
ggsave(filename = "chunky_median_diff_perc.png",
       plot = diff_box, 
       width = 10, height = 7, units = "in", 
       dpi=300, device = "png",
       path = plot_dir)


