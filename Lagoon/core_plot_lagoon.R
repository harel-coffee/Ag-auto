library(ggmap)
library(ggpubr)
library(lubridate)
library(purrr)
library(scales)
library(tidyverse)
# library(ggplot2)

options(digits=9)
options(digits=9)
############################################################
#
#                         Functions
#
############################################################
############################################################
############################################################
plot_fractions_separately <- function(dt, y_lab, tgt_col){
  month_order <- c(9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8)
  for (mon in month_order){
    curr_month <- dt %>% filter(month==mon) %>% data.table()
    if (mon != 9){
      y_lab <- ""
    }
    if (tgt_col == "rain_fraction"){
       assign(x = paste0("month_", mon),
              value ={frac_rain(dt=curr_month, y_lab, tgt_col)})
      } else {
       assign(x = paste0("month_", mon),
              value ={frac_snow(dt=curr_month, y_lab, tgt_col)})
    }
  }
  plt <- ggarrange(plotlist = list(month_9, month_10, month_11, month_12,
                                   month_1, month_2, month_3, month_4, month_5,
                                   month_6, month_7, month_8),
                   ncol = 12, nrow = 1)
  return(plt)
}

frac_rain <- function(dt, y_lab, tgt_col){
  ### This function is meant to be used for 
  ### data with 1 emission and 1 cluster type
  ###
  ###
  dt <- na.omit(dt)
  dt <- subset(dt, select=c("location", "time_period", "emission", "month",
                            "cluster", tgt_col))
  dt <- month_numeric_2_str(dt)
  dt <- dt %>% filter(time_period != "2006-2025") %>% data.table()
  if (length(unique(dt$time_period)) == 3){
    color_ord = c("dodgerblue2", "olivedrab4", "gold")
    } else if (length(unique(dt$time_period)) == 4){
    color_ord = c("grey47", "dodgerblue2", "olivedrab4", "gold")
    } else if (length(unique(dt$time_period)) == 5){
    color_ord = c("red", "grey47", "dodgerblue2", "olivedrab4", "gold")
  }
  time_lbl <- sort(unique(dt$time_period))
  dt$time_period <- factor(dt$time_period, levels=time_lbl)
  suppressWarnings({dt <- within(dt, remove(year))})

  medians <- data.frame(dt) %>% 
             group_by(cluster, time_period, emission, month) %>% 
             summarise( med = median(get(tgt_col))) %>% 
             data.table()

  melted <- melt(dt, id = c("location", "month", "time_period", "emission", "cluster"))
  rm(dt)

  ax_txt_size <- 8; ax_ttl_size <- 12; box_width = 0.53
  the <- theme(plot.margin = unit(c(t=.1, r=0.1, b=.1, l=0), "cm"),
               panel.border = element_rect(fill=NA, size=.3),
               panel.grid.major = element_line(size = 0.05),
               panel.grid.minor = element_blank(),
               panel.spacing = unit(.35, "line"), panel.spacing.y = unit(.5, 'line'),
               legend.title = element_blank(),
               legend.position = "none",
               # legend.position = "bottom", legend.spacing.x = unit(.1, 'line'),
               # legend.key.size = unit(1, "line"), legend.title = element_blank(),
               # legend.text = element_text(size = ax_ttl_size, face="bold"),
               # legend.margin = margin(t=.1, r=0, b=0, l=0, unit = 'line'),
               plot.title = element_text(size=8, face = "bold",
                                         margin = margin(t=.15, r=.1, b=0, l=0, "line")),
               plot.subtitle = element_text(face = "bold"),
               strip.text.x = element_text(size = ax_ttl_size, face = "bold",
                                           margin = margin(.15, 0, .15, 0, "line")),
               strip.text.y = element_text(size = ax_ttl_size, face = "bold",
                                           margin = margin(.15, 0, .15, 0, "line")),
               axis.ticks = element_line(size = .1, color = "black"),
               axis.text.y = element_text(size = ax_txt_size, face = "bold", color = "black", angle = 90),
               axis.text.x = element_text(size = ax_txt_size, face = "bold", color="black",
                                          margin=margin(t=.05, r=5, l=5, b=0,"pt")),
               axis.title.y = element_text(size = ax_ttl_size, face = "bold", 
                                           margin = margin(t=0, r=2, b=0, l=0)),
               axis.title.x = element_blank()
              )
  
  if (length(unique(melted$time_period)) == 3){
    box_p <- ggplot(data = melted, 
                  aes(x=month, y=value, fill=time_period)) +
             the + ylab(y_lab) +
             geom_hline(yintercept= 0, color = "red", size=.3) +
             geom_boxplot(outlier.size=-0.3, notch=F, 
                          width = box_width, lwd=.1, 
                          position = position_dodge(0.6)) +
             facet_grid(~ emission ~ cluster, scales="free") +
             scale_fill_manual(values = color_ord, name = "time\nperiod", 
                               labels = time_lbl) + 
             geom_text(data = medians, 
                       aes(label = sprintf("%1.0f", medians$med), y = medians$med), 
                       size = 2.5, vjust = -.4, position = position_dodge(.6))
                       # + coord_cartesian(ylim = c(boxplot.stats(melted$value)$stats[c(1)])*0.95, 100)
    } else {
      box_p <- ggplot(data = melted, 
                      aes(x=month, y=value, fill=time_period)) +
               the + ylab(y_lab) +
               geom_hline(yintercept= 0, color = "red", size=.3) +
               geom_boxplot(outlier.size=- 0.3, notch=F, width = box_width, lwd=.1, 
                            position = position_dodge(0.6)) +
               facet_grid(~ emission ~ cluster, scales="free") +
               scale_fill_manual(values = color_ord,
                                 name = "time\nperiod", 
                                 labels = time_lbl) 
               # + coord_cartesian(ylim = c(boxplot.stats(melted$value)$stats[c(1)]*.95, 100))
  }       
  return(box_p)
}

frac_snow <- function(dt, y_lab, tgt_col){
  ### This function is meant to be used for 
  ### data with 1 emission and 1 cluster type
  dt <- na.omit(dt)
  dt <- subset(dt, select=c("location", "time_period", "emission", "month",
                            "cluster", tgt_col))
  dt <- month_numeric_2_str(dt)
  dt <- dt %>% filter(time_period != "2006-2025") %>% data.table()
  if (length(unique(dt$time_period)) == 3){
    color_ord = c("dodgerblue2", "olivedrab4", "gold")
    } else if (length(unique(dt$time_period)) == 4){
    color_ord = c("grey47", "dodgerblue2", "olivedrab4", "gold")
    } else if (length(unique(dt$time_period)) == 5){
    color_ord = c("red", "grey47", "dodgerblue2", "olivedrab4", "gold")
  }
  time_lbl <- sort(unique(dt$time_period))
  dt$time_period <- factor(dt$time_period, levels=time_lbl)
  suppressWarnings({dt <- within(dt, remove(year))})

  medians <- data.frame(dt) %>% 
             group_by(cluster, time_period, emission, month) %>% 
             summarise( med = median(get(tgt_col))) %>% 
             data.table()

  melted <- melt(dt, id = c("location", "month", "time_period", "emission", "cluster"))
  rm(dt)

  ax_txt_size <- 8; ax_ttl_size <- 12; box_width = 0.53
  the <- theme(plot.margin = unit(c(t=.1, r=0.1, b=.1, l=0), "cm"),
               panel.border = element_rect(fill=NA, size=.3),
               panel.grid.major = element_line(size = 0.05),
               panel.grid.minor = element_blank(),
               panel.spacing = unit(.35, "line"),
               panel.spacing.y = unit(.5, 'line'),
               legend.title = element_blank(),
               legend.position = "none",
               # legend.position = "bottom", 
               # legend.key.size = unit(1, "line"), legend.spacing.x = unit(.1, 'line'),
               # legend.text = element_text(size = ax_ttl_size, face="bold"),
               # legend.margin = margin(t=.1, r=0, b=0, l=0, unit = 'line'),
               # legend.title = element_blank(),
               plot.title = element_text(size=8, face = "bold",
                                         margin = margin(t=.15, r=.1, b=0, l=0, "line")),
               plot.subtitle = element_text(face = "bold"),
               strip.text.x = element_text(size = ax_ttl_size, face = "bold",
                                           margin = margin(.15, 0, .15, 0, "line")),
               strip.text.y = element_text(size = ax_ttl_size, face = "bold",
                                           margin = margin(.15, 0, .15, 0, "line")),
               axis.ticks = element_line(size = .1, color = "black"),
               axis.text.y = element_text(size = ax_txt_size, face = "bold", color = "black", angle = 90),
               axis.text.x = element_text(size = ax_txt_size, face = "bold", color="black",
                                          margin=margin(t=.05, r=5, l=5, b=0,"pt")),
               axis.title.y = element_text(size = ax_ttl_size, face = "bold", 
                                           margin = margin(t=0, r=2, b=0, l=0)),
               axis.title.x = element_blank()
              )
  
  if (length(unique(melted$time_period)) == 3){
    box_p <- ggplot(data = melted, aes(x=month, y=value, fill=time_period)) +
             the + ylab(y_lab) +
             geom_hline(yintercept= 0, color = "red", size=.3) +
             geom_boxplot(outlier.size=-0.3, notch=F, width = box_width, lwd=.1, 
                          position = position_dodge(0.6)) +
             facet_grid(~ emission ~ cluster, scales="free") +
             scale_fill_manual(values = color_ord, name = "time\nperiod", 
                               labels = time_lbl) + 
             geom_text(data = medians, y = medians$med, 
                       aes(label = sprintf("%1.0f", medians$med)), 
                       size = 2.5, vjust = -.4, position = position_dodge(.6))
             # + coord_cartesian(ylim = c(0, boxplot.stats(melted$value)$stats[c(5)])*1.05)

    } else {
      box_p <- ggplot(data = melted, aes(x=month, y=value, fill=time_period)) +
               the + ylab(y_lab) +
               geom_hline(yintercept= 0, color = "red", size=.3) +
               geom_boxplot(outlier.size=-0.3, notch=F, width = box_width, lwd=.1, 
                            position = position_dodge(0.6)) +
               facet_grid(~ emission ~ cluster, scales="free") +
               scale_fill_manual(values = color_ord, name = "time\nperiod", 
                                 labels = time_lbl)
               # + coord_cartesian(ylim = c(0, boxplot.stats(melted$value)$stats[c(5)])*1.05)
  }
  if (tgt_col == "perc_diff"){
    box_p <- box_p + geom_hline(yintercept= 0, color = "red", size=.2)
  }
          
  return(box_p)
}
############################################################
seasonal_fraction_season_x <-function(data_tb,y_lab="rain fraction (%)",tgt_col="rain_fraction"){
  data_tb$rain_fraction <- data_tb$rain_fraction * 100
  data_tb$snow_fraction <- data_tb$snow_fraction * 100
  if (tgt_col=="rain_fraction"){
     data_tb <- within(data_tb, remove(model, year, location, 
                                       seasonal_cum_precip, snow_fraction))
     } else {
       data_tb <- within(data_tb, remove(model, year, location, 
                                         seasonal_cum_precip, rain_fraction))
  }
  season_levels <- c("fall", "winter", "spring", "summer")
  data_tb$season <- factor(data_tb$season, levels=season_levels, order=T)
  
  medians <- data.frame(data_tb) %>% 
             group_by(cluster, time_period, emission, season) %>% 
             summarise(med = median(get(tgt_col))) %>% 
             data.table()

  melted <- melt(data_tb, id = c("emission", "time_period", "cluster", "season"))
  rm(data_tb)
  
  time_label <- sort(unique(melted$time_period))
  if (length(unique(melted$time_period)) == 3){
     color_ord = c("dodgerblue2", "olivedrab4", "gold")
     } else if (length(unique(melted$time_period)) == 4){
       color_ord = c("grey47", "dodgerblue2", "olivedrab4", "gold")
     } else if (length(unique(melted$time_period)) == 5){
     color_ord = c("red", "grey47", "dodgerblue2", "olivedrab4", "gold")
  }

  categ_label <- c("Western coastal", "Cascade foothills", "Northwest Cascades", 
                   "Northcentral Cascades", "Northeast Cascades")
  melted$cluster <- factor(melted$cluster, levels=categ_label)
  melted$time_period <- factor(melted$time_period, levels=time_label)
  
  ax_txt_size <- 8; ax_ttl_size <- 10; box_width = 0.6
  ax_txt_size <- 6; ax_ttl_size <- 8; box_width = 0.6

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
               legend.margin = margin(t=-.2, r=0, b=0, l=0, unit = 'line'),
               legend.title = element_blank(),
               plot.title = element_text(size=8, face = "bold",
                                         margin = margin(t=.15, r=.1, b=0, l=0, "line")),
               plot.subtitle = element_text(size=ax_txt_size, face = "plain"),
               strip.text.x = element_text(size = ax_ttl_size, face = "bold",
                                           margin = margin(.15, 0, .15, 0, "line")),
               axis.ticks = element_line(size = .1, color = "black"),
               axis.text.y = element_text(size = ax_txt_size, face = "bold", 
                                          color = "black"),
               axis.text.x = element_text(size = ax_txt_size, face = "bold", 
                                          color="black",
                                          margin=margin(t=.05, r=5, l=5, b=0,"pt")
                                          ),
               axis.title.y = element_text(size = ax_ttl_size, face = "bold", 
                                           margin = margin(t=0, r=2, b=0, l=0)),
               axis.title.x = element_blank()
              )

  signif <- if (grepl("diff", tgt_col)) "%1.1f" else "%1.0f"
  ########################################################################
  ######
  ###### plot
  ######
  ggplot(data = melted, aes(x=season, y=value, fill=time_period)) +
  the + 
  # geom_hline(yintercept= 0, color = "red", size=.3)+
  geom_boxplot(outlier.size = - 0.3, notch=F, 
               width = box_width, lwd=.1, 
               position = position_dodge(0.8), outlier.shape=NA
               ) +
  scale_x_discrete(expand=c(0.1, 0)) + 
  # labs(x="", y="") + # theme_bw() + 
  # facet_grid(~ emission, scales="free") +
  xlab("precip. group") +
  ylab(y_lab) + 
  # ylim(quantile(melted$value, probs = c(0.05, 0.95))) + 
  scale_fill_manual(values = color_ord, labels = time_label) +
  geom_text(data = medians, 
            aes(label = sprintf(signif, medians$med), y = medians$med), 
            size = 2, fontface = "bold",
            position = position_dodge(.8), vjust = -.6)
}
####
seasonal_fraction_clust_x <-function(data_tb,y_lab="rain fraction (%)",tgt_col="rain_fraction"){
  data_tb$rain_fraction <- data_tb$rain_fraction * 100
  data_tb$snow_fraction <- data_tb$snow_fraction * 100
  if (tgt_col=="rain_fraction"){
     data_tb <- within(data_tb, remove(model, year, location, 
                                       seasonal_cum_precip, snow_fraction))
     } else {
       data_tb <- within(data_tb, remove(model, year, location, 
                                         seasonal_cum_precip, rain_fraction))
  }

  region_levels <- c("Western coastal", "Cascade foothills", 
                     "Northwest Cascades", "Northcentral Cascades", 
                     "Northeast Cascades")
  data_tb$cluster <- factor(data_tb$cluster, levels=region_levels, order=T)
  medians <- data.frame(data_tb) %>% 
             group_by(cluster, time_period, emission, season) %>% 
             summarise(med = median(get(tgt_col))) %>% 
             data.table()

  melted <- melt(data_tb, id = c("emission", "time_period", 
                                 "cluster", "season"))
  rm(data_tb)
  
  time_label <- sort(unique(melted$time_period))
  if (length(unique(melted$time_period)) == 3){
     color_ord = c("dodgerblue2", "olivedrab4", "gold")
     } else if (length(unique(melted$time_period)) == 4){
       color_ord = c("grey47", "dodgerblue2", "olivedrab4", "gold")
     } else if (length(unique(melted$time_period)) == 5){
     color_ord = c("red", "grey47", "dodgerblue2", "olivedrab4", "gold")
  }

  categ_label <-  c("Western coastal", "Cascade foothills", 
                   "Northwest Cascades", "Northcentral Cascades", 
                   "Northeast Cascades")
  melted$cluster <- factor(melted$cluster, levels=categ_label)
  melted$time_period <- factor(melted$time_period, levels=time_label)
  
  ax_txt_size <- 8; ax_ttl_size <- 10; box_width = 0.6
  ax_txt_size <- 6; ax_ttl_size <- 8; box_width = 0.6

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
               legend.margin = margin(t=-.2, r=0, b=0, l=0, unit = 'line'),
               legend.title = element_blank(),
               plot.title = element_text(size=8, face = "bold",
                                         margin = margin(t=.15, r=.1, b=0, l=0, "line")), # b=-1.5
               plot.subtitle = element_text(size=ax_txt_size, face = "plain"),
               strip.text.x = element_text(size = ax_ttl_size, face = "bold",
                                           margin = margin(.15, 0, .15, 0, "line")),
               axis.ticks = element_line(size = .1, color = "black"),
               axis.text.y = element_text(size = ax_txt_size, face = "bold", 
                                          color = "black"),
               axis.text.x = element_text(size = ax_txt_size, face = "bold", 
                                          color="black",
                                          margin=margin(t=.05, r=5, l=5, b=0,"pt")
                                          ),
               axis.title.y = element_text(size = ax_ttl_size, face = "bold", 
                                           margin = margin(t=0, r=2, b=0, l=0)),
               axis.title.x = element_blank()
              )

  signif <- if (grepl("diff", tgt_col)) "%1.1f" else "%1.0f"
  ########################################################################
  ######
  ###### plot
  ######
  ggplot(data = melted, aes(x=cluster, y=value, fill=time_period)) +
  the +
  geom_boxplot(outlier.size = - 0.3, notch=F, 
               width = box_width, lwd=.1, 
               position = position_dodge(0.8), outlier.shape=NA
               ) +
  scale_x_discrete(expand=c(0.1, 0)) + 
  # labs(x="", y="") + # theme_bw() + 
  facet_grid(~ emission, scales="free") +
  xlab("precip. group") +
  ylab(y_lab) + 
  # ylim(quantile(melted$value, probs = c(0.05, 0.95))) + 
  scale_fill_manual(values = color_ord, labels = time_label) +
  geom_text(data = medians, 
            aes(label = sprintf(signif, medians$med), y = medians$med), 
            size = 2, fontface = "bold",
            position = position_dodge(.8), vjust = -.6)
  # + coord_cartesian(ylim = (boxplot.stats(melted$value)$stats[c(1, 5)])*1.05)
}
########################
annual_fraction <-function(data_tb,y_lab="rain fraction (%)",tgt_col="rain_fraction"){
  data_tb$rain_fraction <- data_tb$rain_fraction * 100
  data_tb$snow_fraction <- data_tb$snow_fraction * 100
  if (tgt_col=="rain_fraction"){
     data_tb <- within(data_tb, remove(model, year, location, 
                                       annual_cum_precip, snow_fraction))
     } else {
       data_tb <- within(data_tb, remove(model, year, location, 
                                         annual_cum_precip, rain_fraction))
   }

  region_levels <-  c("Western coastal", "Cascade foothills", 
                   "Northwest Cascades", "Northcentral Cascades", 
                   "Northeast Cascades")
  data_tb$cluster <- factor(data_tb$cluster, levels=region_levels, order=T)
  medians <- data.frame(data_tb) %>% 
             group_by(cluster, time_period, emission) %>% 
             summarise(med = median(get(tgt_col))) %>% 
             data.table()

  melted <- melt(data_tb, id = c("emission", "time_period", "cluster"))
  rm(data_tb)
  
  time_label <- sort(unique(melted$time_period))
  if (length(unique(melted$time_period)) == 3){
     color_ord = c("dodgerblue2", "olivedrab4", "gold")
     } else if (length(unique(melted$time_period)) == 4){
       color_ord = c("grey47", "dodgerblue2", "olivedrab4", "gold")
     } else if (length(unique(melted$time_period)) == 5){
     color_ord = c("red", "grey47", "dodgerblue2", "olivedrab4", "gold")
  }

  melted$cluster <- factor(melted$cluster, levels=region_levels)
  melted$time_period <- factor(melted$time_period, levels=time_label)
  
  ax_txt_size <- 8; ax_ttl_size <- 10; box_width = 0.6
  ax_txt_size <- 6; ax_ttl_size <- 8; box_width = 0.6

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
               legend.margin = margin(t=-0.2, r=0, b=0, l=0, unit = 'line'),
               legend.title = element_blank(),
               plot.title = element_text(size=8, face = "bold",
                                         margin = margin(t=.15, r=.1, b=0, l=0, "line")), # b=-1.5
               plot.subtitle = element_text(size=ax_txt_size, face = "plain"),
               strip.text.x = element_text(size = ax_ttl_size, face = "bold",
                                           margin = margin(.15, 0, .15, 0, "line")),
               axis.ticks = element_line(size = .1, color = "black"),
               axis.text.y = element_text(size = ax_txt_size, face = "bold", 
                                          color = "black"),
               axis.text.x = element_text(size = ax_txt_size, face = "bold", 
                                          color="black",
                                          margin=margin(t=.05, r=5, l=5, b=0,"pt")
                                          ),
               axis.title.y = element_text(size = ax_ttl_size, face = "bold", 
                                           margin = margin(t=0, r=2, b=0, l=0)),
               axis.title.x = element_blank()
              )

  signif <- if (grepl("diff", tgt_col)) "%1.1f" else "%1.0f"
  ########################################################################
  ######
  ###### plot
  ######
  ggplot(data = melted, aes(x=cluster, y=value, fill=time_period)) +
  the + 
  # geom_hline(yintercept= 0, color = "red", size=.3) +
  geom_boxplot(outlier.size = - 0.3, notch=F, 
               width = box_width, lwd=.1, 
               position = position_dodge(0.8), outlier.shape=NA
               ) +
  scale_x_discrete(expand=c(0.1, 0)) + 
  # labs(x="", y="") + # theme_bw() + 
  # facet_grid(~ emission, scales="free") +
  xlab("precip. group") +
  ylab(y_lab) +
  # ylim(quantile(melted$value, probs = c(0.05, 0.95))) +
  scale_fill_manual(values = color_ord, labels = time_label) +
  geom_text(data = medians, 
            aes(label = sprintf(signif, medians$med), y = medians$med), 
            size = 2, fontface = "bold",
            position = position_dodge(.8), vjust = -.6)
  # + coord_cartesian(ylim = (boxplot.stats(melted$value)$stats[c(1, 5)])*1.05)
}
############################################################
############################################################
############################################################

seasonal_cum_box_clust_x <- function(dt, y_lab, tgt_col, ttl, subttl){
  season_levels <- c("fall", "winter", "spring", "summer")
  dt$season <- factor(dt$season, levels=season_levels, order=T)
  
  region_levels <- c("Western coastal", "Cascade foothills", 
                     "Northwest Cascades", "Northcentral Cascades", 
                     "Northeast Cascades")
  
  dt$cluster <- factor(dt$cluster, levels=region_levels, order=T)
  dt <- dt %>% filter(time_period != "2006-2025") %>% data.table()
  
  dt <- subset(dt, select=c("time_period", 
                            "emission", "season",
                            "cluster", tgt_col))
  
  medians <- data.frame(dt) %>% 
             group_by(cluster, time_period, emission, season) %>% 
             summarise(med = median(get(tgt_col))) %>% 
             data.table()

  melted <- melt(dt, id = c("emission", "season", 
                            "time_period", "cluster")); rm(dt)
  time_label <- sort(unique(melted$time_period))
  
  if (length(unique(melted$time_period)) == 3){
    color_ord = c("dodgerblue2", "olivedrab4", "gold")
    } else if (length(unique(melted$time_period)) == 4){
      color_ord = c("grey47", "dodgerblue2", "olivedrab4", "gold")
    } else if (length(unique(melted$time_period)) == 5){
    color_ord = c("red", "grey47", "dodgerblue2", "olivedrab4", "gold")
  }

  melted$time_period <- factor(melted$time_period, levels=time_label)
  
  ax_txt_size <- 8; ax_ttl_size <- 10; box_width = 0.6
  ax_txt_size <- 6; ax_ttl_size <- 8; box_width = 0.6

  the <- theme(plot.margin = unit(c(t=.1, r=.2, b=.1, l=0.2), "cm"),
               panel.border = element_rect(fill=NA, size=.3),
               panel.grid.major = element_line(size = 0.05),
               panel.grid.minor = element_blank(),
               panel.spacing = unit(.35, "line"),
               panel.spacing.y = unit(.5, 'line'),
               legend.position = "bottom", 
               legend.key.size = unit(.8, "line"),
               legend.spacing.x = unit(.1, 'line'),
               legend.text = element_text(size = ax_ttl_size, face="bold"),
               legend.margin = margin(t=0, r=0, b=0, l=0, unit = 'line'),
               legend.title = element_blank(),
               plot.title = element_text(size = 8, face = "bold",
                                         margin = margin(t=.15, r=.1, 
                                                         b=0, l=0, "line")),
               plot.subtitle = element_text(size=ax_txt_size, face = "plain"),
               strip.text.x = element_text(size = ax_ttl_size, face = "bold",
                                           margin = margin(.15, 0, .15, 0, "line")),
               axis.ticks = element_line(size = .1, color = "black"),
               axis.text.y = element_text(size = ax_txt_size, 
                                          face = "bold", color = "black"),
               axis.text.x = element_text(size = ax_txt_size, 
                                          face = "bold", color="black",
                                          margin=margin(t=.05, r=5, l=5, b=-2,"pt")
                                          ),
               axis.title.y = element_text(size = ax_ttl_size, 
                                           face = "bold", 
                                           margin = margin(t=0, r=4, b=0, l=0)),
               axis.title.x = element_blank()
              )
  signif <- if (grepl("diff", tgt_col)) "%1.1f" else "%1.0f"
  ########
  ########    PLOT
  ########
  bx <- ggplot(data = melted, aes(x=cluster, y=value, fill=time_period)) +
        the + 
        geom_boxplot(outlier.size=-0.3, notch=F, 
                     width = box_width, lwd=.1, 
                     position = position_dodge(0.8), outlier.shape=NA
                     ) +
        scale_x_discrete(expand=c(0.1, 0)) + 
        # labs(x="", y="") + # theme_bw() + 
        facet_grid(~ emission, scales="free") +
        ylab(y_lab) +
        # ylim(quantile(melted$value, probs = c(0.05, 0.95))) +  
        scale_fill_manual(values = color_ord, labels = time_label) +
        geom_text(data = medians, 
                  aes(label = sprintf(signif, medians$med), y=medians$med), 
                  size = 2, fontface = "bold",
                  position = position_dodge(.8), vjust = -.3)
  if(tgt_col == "perc_diff"){
    bx <- bx + geom_hline(yintercept = 0, color = "red", size=.3)
  }
  return(bx)
}

seasonal_cum_box_season_x <- function(dt, y_lab, tgt_col, ttl, subttl){
  season_levels <- c("fall", "winter", "spring", "summer")
  dt$season <- factor(dt$season, levels=season_levels, order=T)
  dt <- dt %>% filter(time_period != "2006-2025") %>% data.table()
  dt <- subset(dt, select=c("time_period", "emission", "season","cluster", tgt_col))
  
  medians <- data.frame(dt) %>% 
             group_by(cluster, time_period, emission, season) %>% 
             summarise(med = median(get(tgt_col))) %>% 
             data.table()

  melted <- melt(dt, id = c("emission", "season", "time_period", "cluster")); rm(dt)
  time_label <- sort(unique(melted$time_period))
  if (length(unique(melted$time_period)) == 3){
    color_ord = c("dodgerblue2", "olivedrab4", "gold")
    } else if (length(unique(melted$time_period)) == 4){
      color_ord = c("grey47", "dodgerblue2", "olivedrab4", "gold")
    } else if (length(unique(melted$time_period)) == 5){
    color_ord = c("red", "grey47", "dodgerblue2", "olivedrab4", "gold")
  }
   categ_label <- c("Western coastal", "Cascade foothills", 
                   "Northwest Cascades", "Northcentral Cascades", 
                   "Northeast Cascades")
  melted$cluster <- factor(melted$cluster, levels=categ_label)
  melted$time_period <- factor(melted$time_period, levels=time_label)
  ax_txt_size <- 8; ax_ttl_size <- 10; box_width = 0.6
  ax_txt_size <- 6; ax_ttl_size <- 8; box_width = 0.6

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
               legend.margin = margin(t=0, r=0, b=0, l=0, unit = 'line'),
               legend.title = element_blank(),
               plot.title = element_text(size=8, face = "bold",
                                         margin = margin(t=.15, r=.1, b=0, l=0, "line")),
               plot.subtitle = element_text(size=ax_txt_size, face = "plain"),
               strip.text.x = element_text(size = ax_ttl_size, face = "bold",
                                           margin = margin(.15, 0, .15, 0, "line")),
               axis.ticks = element_line(size = .1, color = "black"),
               axis.text.y = element_text(size = ax_txt_size, 
                                          face = "bold", color = "black"),
               axis.text.x = element_text(size = ax_txt_size, 
                                          face = "bold", color="black",
                                          margin=margin(t=.05, r=5, l=5, b=-2,"pt")
                                          ),
               axis.title.y = element_text(size = ax_ttl_size, face = "bold", 
                                           margin = margin(t=0, r=4, b=0, l=0)),
               axis.title.x = element_blank()
              )
  signif <- if (grepl("diff", tgt_col)) "%1.1f" else "%1.0f"
  ########
  ########    PLOT
  ########
  bx <- ggplot(data = melted, aes(x=season, y=value, fill=time_period)) +
        the + 
        geom_boxplot(outlier.size = - 0.3, notch=F, 
                     width = box_width, lwd=.1, 
                     position = position_dodge(0.8), outlier.shape=NA) +
        scale_x_discrete(expand=c(0.1, 0)) + 
        # labs(x="", y="") + # theme_bw() + 
        # facet_grid(~ emission, scales="free") +
        ylab(y_lab) + 
        # ylim(quantile(melted$value, probs = c(0.05, 0.95))) +
        scale_fill_manual(values = color_ord, labels = time_label) +
        geom_text(data = medians, 
                  aes(label = sprintf(signif, medians$med), y=medians$med), 
                  size = 2, fontface = "bold",
                  position = position_dodge(.8), vjust = -.3)
  if (tgt_col == "perc_diff"){
    bx <- bx + geom_hline(yintercept= 0, color = "red", size=.3)
  }
  return(bx)
  # + coord_cartesian(ylim = (boxplot.stats(melted$value)$stats[c(1, 5)])*1.05)
}

#
# change the name and maintain the code so it is 
# funtional for wide range of data.
# lets say one column is location, and one column determines
# color of stuff on the map.
#
geo_map_perc_diff <- function(dt_dt, col_col, color_limit){
  x <- sapply(dt_dt$location, 
              function(x) strsplit(x, "_")[[1]], 
              USE.NAMES=FALSE)
  lat <- as.numeric(x[1, ]); long <- as.numeric(x[2, ])
  dt_dt$lat <- lat; dt_dt$long <- long;

  WA_counties <- map_data("county", "washington")
  WA_counties <- WA_counties %>% 
                 filter(subregion %in% c("whatcom", "skagit", "snohomish", "island"
                                         # , "okanogan", "chelan"
                                         ))%>% data.table()
  dt_dt %>%
  ggplot() +
  geom_polygon(data = WA_counties, 
               aes(x = long, y = lat, group = group),
               fill = "grey", color = "black") +
  geom_polygon(data = WA_counties, 
               aes(x=long, y=lat, group = group), 
               fill = NA, colour = "black", size=.5) + 
  geom_point(aes_string(x="long", y="lat", color=col_col), 
             alpha=1, size=2.5) +
  guides(fill = guide_colourbar(barwidth=.1, barheight=2,
         direction = "vertical")) +
  # scale_color_viridis_c(option = "plasma", 
  #                       name = "storm", direction = -1,
  #                       limits = c(color_limit[1], color_limit[2]),
  #                       breaks = pretty_breaks(n=8)) +
  scale_color_gradient2(midpoint = 0, mid = "white", 
                        high = muted("blue"), low = muted("red"), 
                        guide = "colourbar", space = "Lab",
                        limit = c(color_limit[1], color_limit[2]),
                        breaks = c(-200, -100, -50, -25, -20, -15, -10, -5, 0,
                                    5, 10, 15, 20, 25, 50, 100, 200)) + 
  theme(axis.title.y = element_blank(),
        axis.title.x = element_blank(),
        axis.ticks.y = element_blank(), 
        axis.ticks.x = element_blank(),
        axis.text = element_blank(),
        plot.title=element_text(size=14, face="bold"),
        # margin = margin(t=.15, r=.1, b=-.2, l=0, "line")),
        legend.text = element_text(size=8, face="plain"),
        legend.title = element_blank(),
        legend.position = c(0.93, 0.2),
        strip.text = element_text(size=14, face="bold"))
}

Nov_Dec_cum_box <- function(dt, y_lab, tgt_col){

  # suppressWarnings({dt <- within(dt, remove(day, precip, model))})
  # if ("evap" %in% colnames(dt)){
  #   suppressWarnings({dt <- within(dt, remove(evap, runoff, 
  #                                             base_flow, run_p_base))})
  # }

  dt <- subset(dt, select=c("time_period", # "location",
                            "emission", "month",
                            "cluster", tgt_col))
  # dt <- cluster_numeric_2_str(dt); 
  dt <- month_numeric_2_str(dt)

  dt <- dt %>% 
        filter(# time_period != "1950-2005" & 
               time_period != "2006-2025") %>% 
        data.table()

  time_label <- sort(unique(dt$time_period))
  dt$time_period <- factor(dt$time_period, levels=time_label)
  # suppressWarnings({dt <- within(dt, remove(year))})

  medians <- data.frame(dt) %>% 
             group_by(cluster, time_period, emission, month) %>% 
             summarise( med = median(get(tgt_col))) %>% 
             data.table()
  melted <- melt(dt, id = c("month", # "location", 
                            "time_period", "emission",
                            "cluster"))
  
  if (length(unique(melted$time_period)) == 3){
    color_ord = c("dodgerblue2", "olivedrab4", "gold")
    } else if (length(unique(melted$time_period)) == 4){
      color_ord = c("grey47", "dodgerblue2", "olivedrab4", "gold")
    } else if (length(unique(melted$time_period)) == 5){
    color_ord = c("red", "grey47", "dodgerblue2", "olivedrab4", "gold")
  }
  rm(dt)
  ax_txt_size <- 8; ax_ttl_size <- 10; box_width = 0.7
  the <- theme(plot.margin = unit(c(t=.1, r=.2, b=.1, l=0.2), "cm"),
               panel.border = element_rect(fill=NA, size=.3),
               panel.grid.major = element_line(size = 0.05),
               panel.grid.minor = element_blank(),
               panel.spacing = unit(.35, "line"),
               legend.position = "bottom", 
               legend.key.size = unit(1.5, "line"),
               legend.spacing.x = unit(.1, 'line'),
               panel.spacing.y = unit(.5, 'line'),
               legend.text = element_text(size = ax_ttl_size, face="bold"),
               legend.margin = margin(t=-.2, r=0, b=0, l=0, unit = 'line'),
               legend.title = element_blank(),
               plot.title = element_text(size = ax_ttl_size, face = "bold"),
               plot.subtitle = element_text(face = "bold"),
               strip.text.x = element_text(size = ax_ttl_size, face = "bold",
                                           margin = margin(.15, 0, .15, 0, "line")),
               strip.text.y = element_text(size = ax_ttl_size, face = "bold",
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
  signif <- if (grepl("diff", tgt_col)) "%1.1f" else "%1.0f"
  
  ggplot(data = melted, 
        aes(x=month, y=value, fill=time_period)) +
  the + geom_hline(yintercept= 0, color = "red", size=.3) +
  geom_boxplot(outlier.size = -0.3, notch=F, 
               width = box_width, lwd=.1,
               position = position_dodge(.8), outlier.shape=NA
               ) +
  facet_grid(~ cluster ~ emission, scales="free") +
  ylab(y_lab) +
  scale_fill_manual(values = color_ord,
                    name = "time\nperiod",
                    labels = time_label) + 
  geom_text(data = medians, 
            aes(label = sprintf(signif, medians$med), y = medians$med),
            size = 2.5, vjust = -.6, position = position_dodge(.8))
}

Nov_Dec_Diffs <- function(dt, y_lab, tgt_col, ttl, subttl){
  dt <- dt %>% 
        filter(# time_period != "1950-2005" & 
               time_period != "2006-2025") %>% 
        data.table()

  dt <- subset(dt, select=c("time_period", "emission", # "location",
                            "month", "cluster", tgt_col))
  dt <- month_numeric_2_str(dt)
  medians <- data.frame(dt) %>% 
             group_by(cluster, time_period, emission, month) %>%
             summarise(med = median(get(tgt_col))) %>% 
             data.table()

  melted <- melt(dt, id = c("emission", "month", # "location", 
                            "time_period", "cluster"))
  rm(dt)
  
  time_label <- sort(unique(melted$time_period))
  if (length(unique(melted$time_period)) == 3){
    color_ord = c("dodgerblue2", "olivedrab4", "gold")
    } else if (length(unique(melted$time_period)) == 4){
      color_ord = c("grey47", "dodgerblue2", "olivedrab4", "gold")
    } else if (length(unique(melted$time_period)) == 5){
    color_ord = c("red", "grey47", "dodgerblue2", "olivedrab4", "gold")
  }

  categ_label <- c("Western coastal", "Cascade foothills", 
                   "Northwest Cascades", "Northcentral Cascades", 
                   "Northeast Cascades")
  melted$cluster <- factor(melted$cluster, levels=categ_label)
  melted$time_period <- factor(melted$time_period, levels=time_label)
  
  ax_txt_size <- 8; ax_ttl_size <- 10; box_width = 0.5

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
               plot.subtitle = element_text(size=ax_txt_size, face = "plain"),
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

  signif <- if (grepl("diff", tgt_col)) "%1.1f" else "%1.0f"
    
  ########
  ########    PLOT
  ########
  ggplot(data = melted, aes(x=month, y=value, fill=time_period)) +
  the + 
  geom_hline(yintercept= 0, color = "red", size=.3) + 
  geom_boxplot(outlier.size = - 0.3, notch=F, 
               width = box_width, lwd=.1, 
               position = position_dodge(0.8), outlier.shape=NA
               ) +
  scale_x_discrete(expand=c(0.1, 0)) + 
  # labs(x="", y="") + # theme_bw() + 
  facet_grid(~ emission, scales="free") +
  xlab("precip. group") +
  ylab(y_lab) + 
  # ylim(quantile(melted$value, probs = c(0.05, 0.95))) +
  scale_fill_manual(values = color_ord, labels = time_label) +
  geom_text(data = medians, 
            aes(label = sprintf(signif, medians$med), y = medians$med), 
            size = 3, fontface = "bold",
            position = position_dodge(.8), vjust = -.6)
}

#
# Following works on both precip and runoffs
#
ann_wtrYr_chunk_cum_box_cluster_x <- function(dt, y_lab, tgt_col, ttl, subttl){
  # if (tgt_col=="annual_cum_runbase" | tgt_col=="chunk_cum_runbase"){
  #   suppressWarnings({dt <- within(dt, remove(evap, runoff, 
  #     base_flow, run_p_base))})
  # }
  # suppressWarnings({dt <- within(dt, 
  #                                remove(month, day, year, precip, 
  #                                       model, wtr_yr, tmean, rain, snow, 
  #                                       precip, rain_portion))})
  # dt <- cluster_numeric_2_str(dt)

  # toss unwanted time periods
  dt <- dt %>% 
        filter(# time_period != "1950-2005" & 
               time_period != "2006-2025") %>% 
        data.table()
  # if ("diff" %in% colnames(dt)){
  #   dt <- subset(dt, select=c("location", "time_period", "emission",
  #                             "cluster", tgt_col))
  # }

  dt <- subset(dt, select=c("time_period", "emission", # "location",
                            "cluster", tgt_col))
  medians <- data.frame(dt) %>% 
             group_by(cluster, time_period, emission) %>% 
             summarise(med = median(get(tgt_col))) %>% 
             data.table()
  melted <- melt(dt, id = c("emission", # "location", 
                            "time_period", "cluster")); rm(dt)
  time_label <- sort(unique(melted$time_period))
  if (length(unique(melted$time_period)) == 3){
     color_ord = c("dodgerblue2", "olivedrab4", "gold")
     } else if (length(unique(melted$time_period)) == 4){
       color_ord = c("grey47", "dodgerblue2", "olivedrab4", "gold")
     } else if (length(unique(melted$time_period)) == 5){
     color_ord = c("red", "grey47", "dodgerblue2", "olivedrab4", "gold")
  }

   categ_label <- c("Western coastal", "Cascade foothills", 
                    "Northwest Cascades", "Northcentral Cascades", 
                    "Northeast Cascades")
  melted$cluster <- factor(melted$cluster, levels=categ_label)
  melted$time_period <- factor(melted$time_period, levels=time_label)

  ax_txt_size <- 8; ax_ttl_size <- 10; box_width = 0.6
  ax_txt_size <- 6; ax_ttl_size <- 8; box_width = 0.6
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
               legend.margin = margin(t=0, r=0, b=0, l=0, unit = 'line'),
               legend.title = element_blank(),
               plot.title = element_text(size = ax_ttl_size, face = "bold",
                                         margin = margin(t=.15, r=.1, b=0, l=0, "line")),
               plot.subtitle = element_text(size=ax_txt_size, face = "plain"),
               strip.text.x = element_text(size = ax_ttl_size, face = "bold",
                                           margin = margin(.15, 0, .15, 0, "line")),
               axis.ticks = element_line(size = .1, color = "black"),
               axis.text.y = element_text(size = ax_txt_size, face = "bold", 
                                          color = "black"),
               axis.text.x = element_text(size = ax_txt_size, face = "bold", 
                                          color="black",
                                          margin=margin(t=.05, r=5, l=5, b=0,"pt")
                                          ),
               axis.title.y = element_text(size = ax_ttl_size, face = "bold", 
                                           margin = margin(t=0, r=2, b=0, l=0)),
               axis.title.x = element_blank()
              )
  signif <- if (grepl("diff", tgt_col)) "%1.1f" else "%1.0f"
  ########
  ########    PLOT
  ########
  bx <- ggplot(data = melted, aes(x=cluster, y=value, fill=time_period)) +
        the + 
    # geom_hline(yintercept= 0, color = "red", size=.3) +
    geom_boxplot(outlier.size = - 0.3, notch=F, 
                 width = box_width, lwd=.1,
                 position = position_dodge(0.8), outlier.shape=NA) +
    scale_x_discrete(expand=c(0.1, 0)) + 
    # facet_grid(~ emission, scales="free") +
    xlab("precip. group") +
    ylab(y_lab) + 
    scale_fill_manual(values = color_ord, labels = time_label) +
    geom_text(data = medians, 
              aes(label = sprintf(signif, medians$med), y = medians$med), 
              size = 2, fontface = "bold",
              position = position_dodge(.8), vjust = -.3)

  if (tgt_col=="perc_diff"){
   bx <- bx + geom_hline(yintercept= 0, color = "red", size=.3)
  }
   return(bx)  
}

box_trend_monthly_cum <- function(dt, p_type="trend", trend_type="median", y_lab, tgt_col){
  #
  # input p_type is in {box, trend} (box plot or line plot)
  #   trend_type is in {mean, median} (line plot)
  #

  # suppressWarnings({dt <- within(dt, remove(day, precip, model))})
  # if ("evap" %in% colnames(dt)){
  #   suppressWarnings({dt <- within(dt, remove(evap, runoff, 
  #                                            base_flow, run_p_base))})
  # }

  dt <- subset(dt, select=c("location", "time_period", "emission", "month",
                            "cluster", tgt_col))
  # dt <- cluster_numeric_2_str(dt); 
  dt <- month_numeric_2_str(dt)
  
  if (p_type=="box"){
    dt <- dt %>% 
          filter(# time_period != "1950-2005" & 
                 time_period != "2006-2025") %>% 
          data.table()
    if (length(unique(dt$time_period)) == 3){
      color_ord = c("dodgerblue2", "olivedrab4", "gold")
      } else if (length(unique(dt$time_period)) == 4){
      color_ord = c("grey47", "dodgerblue2", "olivedrab4", "gold")
      } else if (length(unique(dt$time_period)) == 5){
      color_ord = c("red", "grey47", "dodgerblue2", "olivedrab4", "gold")
    }
    time_lbl <- sort(unique(dt$time_period))
    dt$time_period <- factor(dt$time_period, levels=time_lbl)
    suppressWarnings({dt <- within(dt, remove(year))})

    medians <- data.frame(dt) %>% 
               group_by(cluster, time_period, emission, month) %>% 
               summarise( med = median(get(tgt_col))) %>% 
               data.table()
    
    melted <- melt(dt,
                   id = c("location", "month", "time_period", 
                          "emission", "cluster"))
    rm(dt)

    ax_txt_size <- 8; ax_ttl_size <- 12; box_width = 0.53
    ax_txt_size <- 6; ax_ttl_size <- 8;
    the <- theme(plot.margin = unit(c(t=.1, r=.2, b=.1, l=0.2), "cm"),
                 panel.border = element_rect(fill=NA, size=.3),
                 panel.grid.major = element_line(size = 0.05),
                 panel.grid.minor = element_blank(),
                 panel.spacing = unit(.35, "line"),
                 legend.position = "bottom", 
                 legend.key.size = unit(.8, "line"),
                 legend.spacing.x = unit(.1, 'line'),
                 panel.spacing.y = unit(.5, 'line'),
                 legend.text = element_text(size=ax_ttl_size, face="bold"),
                 legend.margin = margin(t=-.2, r=0, b=0, l=0, unit = 'line'),
                 legend.title = element_blank(),
                 plot.title = element_text(size = ax_ttl_size, face = "bold",
                                         margin = margin(t=.15, r=.1, b=0, l=0, "line")),
                 plot.subtitle = element_text(face = "bold"),
                 strip.text.x = element_text(size = ax_ttl_size, face = "bold",
                                             margin = margin(.15, 0, .15, 0, "line")),
                 strip.text.y = element_text(size = ax_ttl_size, face = "bold",
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
    
    if (length(unique(melted$time_period)) == 3){
      box_p <- ggplot(data = melted, 
                     aes(x=month, y=value, fill=time_period)) +
               the + 
               # geom_hline(yintercept= 0, color = "red", size=.3) + 
               geom_boxplot(outlier.size = - 0.3, notch=F, 
                            width = box_width, lwd=.1, 
                            position = position_dodge(0.6), outlier.shape=NA) +
               # facet_grid(~ emission, scales="free") +
               ylab(y_lab) +
               # ylim(quantile(melted$value, probs = c(0.05, 0.95))) + 
               scale_fill_manual(values = color_ord,
                                 name = "time\nperiod", 
                                 labels = time_lbl) + 
               geom_text(data = medians, 
                         aes(label = sprintf("%1.0f", medians$med), y = medians$med), 
                         size = 2.5, vjust = -.4, position = position_dodge(.6))
      if (tgt_col == "perc_diff"){
        box_p <- box_p + geom_hline(yintercept= 0, color = "red", size=.3)
      }
              
      } else {
        box_p <- ggplot(data = melted, 
                    aes(x=month, y=value, fill=time_period)) +
                 the + 
                 # geom_hline(yintercept= 0, color = "red", size=.3) + 
                 geom_boxplot(outlier.size = - 0.3, notch=F, 
                              width = box_width, lwd=.1, outlier.shape=NA,
                              position = position_dodge(0.6)) +
                 # facet_grid(~ emission, scales="free") +
                 ylab(y_lab) +
                 scale_fill_manual(values = color_ord, name = "time\nperiod", 
                                   labels = time_lbl)
        if (tgt_col == "perc_diff"){
        box_p <- box_p + geom_hline(yintercept= 0, color = "red", size=.3)
      }
    }            
    return(box_p)

  } else {
    time_lbl <- sort(unique(dt$time_period))
    dt$time_period <- factor(dt$time_period, levels=time_lbl)
  
    if (trend_type=="median"){
      dt <- dt %>%
            group_by(time_period, month, emission, cluster, year) %>%
            summarise(stat_col=median(monthly_cum_precip))%>%
            data.table()
      y_lab <- "medians of cum. monthly precip. (over models)"
      } else {
        dt <- dt %>%
              group_by(time_period, month, emission, cluster, year) %>%
              summarise(stat_col=mean(monthly_cum_precip))%>%
              data.table()
        y_lab <- "means of cum. monthly precip. (over models)"
    }
      
    dt$month <- as.character(dt$month)
    dt$month <- factor(dt$month, 
                       levels=as.character(c(9, 10, 11, 12, 
                                             1, 2, 3, 4, 5, 6, 7, 8)))

    ax_txt_size <- 15; ax_ttl_size <- 17;
    the <- theme(plot.margin = unit(c(t=.1, r=.2, b=.1, l=0.2), "cm"),
                 panel.border = element_rect(fill=NA, size=.3),
                 panel.grid.major = element_line(size = 0.05),
                 panel.grid.minor = element_blank(),
                 legend.position = "bottom", 
                 legend.key.size = unit(1.5, "line"),
                 legend.spacing.x = unit(1, 'line'),
                 panel.spacing.x = unit(2, 'line'),
                 panel.spacing.y = unit(1, 'line'),
                 legend.text = element_text(size = ax_ttl_size, face="bold"),
                 legend.margin = margin(t=.1, r=0, b=0, l=0, unit = 'line'),
                 legend.title = element_blank(),
                 plot.title = element_text(size = ax_ttl_size, face = "bold"),
                 plot.subtitle = element_text(face = "bold"),
                 strip.text.x = element_text(size = ax_ttl_size, face = "bold",
                                             margin = margin(.15, 0, .15, 0, "line")),
                 strip.text.y = element_text(size = ax_ttl_size, face = "bold",
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
                 axis.title.x = element_text(size = ax_ttl_size , face = "bold",
                                             margin = margin(t=2, r=0, b=-10, l=0))
                      )
    line_p <- ggplot(data=dt, 
                     aes(x=year, y=stat_col, group=time_period, color=time_period)) +
              geom_line() +
              the + 
              facet_grid(~ emission ~ cluster ~ month,
                         labeller=labeller(month = month_names), scales="free") +
              ylab(y_lab)

      return(line_p)
  }
}

############################################################
#      
#            STORM
#
#
box_dt_25 <- function(dt_25){
  categ_lab <- sort(unique(dt_25$return_period))
  
  if (length(unique(dt_25$return_period)) == 3){
    color_ord = c("dodgerblue2", "olivedrab4", "gold")
    } else if (length(unique(dt_25$return_period)) == 4){
      color_ord = c("grey47", "dodgerblue2", "olivedrab4", "gold")
    } else if (length(unique(dt_25$return_period)) == 5){
    color_ord = c("red", "grey47", "dodgerblue2", "olivedrab4", "gold")
  }

  medians <- data.frame(dt_25) %>% 
             group_by(return_period, emission, cluster) %>% 
             summarise(med_25 = median(twenty_five_years)) %>% 
             data.table()

  melted <- melt(dt_25, id = c("cluster", "return_period", "emission"))

  ax_txt_size <- 8; ax_ttl_size <- 10; box_width = 0.65
  # ax_txt_size <- 5; ax_ttl_size <- 6; box_width = 0.53
  the <- theme(plot.margin = unit(c(t=.1, r=.2, b=.1, l=0.2), "cm"),
               panel.border = element_rect(fill=NA, size=.3),
               panel.grid.major = element_line(size = 0.05),
               panel.grid.minor = element_blank(),
               panel.spacing = unit(.35, "line"),
               legend.position = "bottom", 
               legend.key.size = unit(0.8, "line"),
               legend.spacing.x = unit(.1, 'line'),
               panel.spacing.y = unit(.5, 'line'),
               legend.text = element_text(size=ax_txt_size, face="bold"),
               legend.margin = margin(t=.1, r=0, b=0, l=0, unit = 'line'),
               legend.title = element_blank(),
               plot.title = element_text(size = ax_ttl_size, face = "bold",
                                         margin = margin(t=.15, r=0.1, 
                                                         b=0, l=0, "line")),
               # plot.subtitle = element_text(size=ax_txt_size, face = "bold"),
               strip.text.x = element_text(size = ax_ttl_size, face = "bold",
                                           margin = margin(.15, 0, .15, 0, "line")),
               axis.ticks = element_blank(), #element_line(size = .1, color = "black"),
               axis.text.y = element_text(size = ax_txt_size, 
                                          face = "bold", color = "black"),
               # axis.text.x = element_text(size = ax_txt_size, 
               #                            face = "bold", color = "black"),
               axis.text.x = element_blank(),
               axis.title.y = element_text(size = ax_ttl_size, 
                                           face = "bold", 
                                           margin = margin(t=0, r=2, b=0, l=0)),
               axis.title.x = element_blank()
              )
  box_p <- ggplot(data = melted, 
                  aes(x=cluster, y=value, fill=return_period)) +
           geom_hline(yintercept= 0, color = "red", size=.3) +
           geom_boxplot(outlier.size = -0.3, notch=F, 
                        width = box_width, lwd=.1, 
                        position = position_dodge(0.85), outlier.shape=NA
                        ) +
           # facet_grid(~ emission) +
           xlab("precip. group") + 
           ylab("design storm intensity (mm/hr)") + 
           # ylim(quantile(melted$value, probs = c(0.05, 0.95))) + 
           scale_fill_manual(values = color_ord,
                             name = "Return\nPeriod", 
                             labels = categ_lab) + 
           scale_y_continuous(breaks = seq(0, 20, by=2)) + 
           the +
           geom_text(data = medians, 
                     aes(label = sprintf("%1.1f", medians$med_25), 
                         y = medians$med_25),
                     size=4, vjust = -.3, fontface="bold",
                     position = position_dodge(.85))
}

storm_diff_box_25yr <- function(data_tb, tgt_col){
  data_tb <- data_tb %>% 
             filter(time_interval=="twenty_five_years") %>% 
             data.table()

  needed_cols <- c("return_period", "emission", "cluster", tgt_col)
  data_tb <- subset(data_tb, select=needed_cols)

  time_label <- sort(unique(data_tb$return_period))
  data_tb$return_period <- factor(data_tb$return_period, levels=time_label)
  data_tb$cluster <- factor(data_tb$cluster, 
                            levels=c("Western coastal", 
                                     "Cascade foothills", 
                                     "Northwest Cascades", 
                                     "Northcentral Cascades", 
                                     "Northeast Cascades"))
     
  if (length(time_label) == 3){
    color_ord = c("dodgerblue2", "olivedrab4", "gold")
    } else if (length(time_label) == 4){
      color_ord = c("grey47", "dodgerblue2", "olivedrab4", "gold")
    } else if (length(time_label) == 5){
    color_ord = c("red", "grey47", "dodgerblue2", "olivedrab4", "gold")
  }
  ax_txt_size <- 8; ax_ttl_size <- 10; box_width = 0.53
  
  the <- theme(plot.margin = unit(c(t=.1, r=.2, b=.1, l=0.2), "cm"),
               panel.border = element_rect(fill=NA, size=.3),
               panel.grid.major = element_line(size = 0.05),
               panel.grid.minor = element_blank(),
               panel.spacing = unit(.35, "line"),
               legend.position = "bottom", 
               legend.key.size = unit(0.8, "line"),
               legend.spacing.x = unit(.1, 'line'),
               panel.spacing.y = unit(.5, 'line'),
               legend.text = element_text(size = ax_txt_size, face="bold"),
               legend.margin = margin(t=.1, r=0, b=0, l=0, unit = 'line'),
               legend.title = element_blank(),
               plot.title = element_text(size = ax_ttl_size, face = "bold", 
                                         margin = margin(t=.15, r=0.1, 
                                                         b=0, l=0, "line")),
               plot.subtitle = element_text(size=ax_txt_size, face = "bold"),
               strip.text.x = element_text(size = ax_ttl_size, face = "bold",
                                           margin = margin(.15, 0, .15, 0, "line")),
               strip.text.y = element_text(size = ax_ttl_size, face = "bold",
                                           margin = margin(.15, 0, .15, 0, "line")),
               axis.ticks = element_blank(), #element_line(size = .1, color = "black"),
               axis.text.y = element_text(size = ax_txt_size, 
                                          face = "bold", color = "black"),
               axis.text.x = element_blank(),
               axis.title.y = element_text(size = ax_ttl_size, 
                                           face = "bold", 
                                           margin = margin(t=0, r=2, b=0, l=0)),
               axis.title.x = element_blank()
              )
  if (tgt_col=="perc_diff"){
     y_labb <- "differences (%)"
     } else {
      y_labb <- "magnitude of differences"
  }
  # box_title <- "diff. of 25 yr/24 hr. design storm"

  medians <- data.frame(data_tb) %>% 
             group_by(return_period, emission, cluster) %>% 
             summarise( med = median(get(tgt_col))) %>% 
             data.table()

  box_p <- ggplot(data = data_tb, 
                  aes(x=cluster, y=get(tgt_col), fill=return_period)) +
           geom_hline(yintercept= 0, color = "red", size=.3) + 
           geom_boxplot(outlier.size = - 0.3, notch=F, 
                        width = box_width, lwd=.1, 
                        position = position_dodge(0.8), outlier.shape=NA
                        ) +
           # labs(x="", y="") + # theme_bw() + 
           # facet_grid(~ emission, scales="free") + # , ncol=4 goes with facet_wrap
           ylab(y_labb) + 
           scale_fill_manual(values = color_ord,
                             name = "Return\nPeriod", 
                             labels = time_label) + 
           the +
           scale_y_continuous(breaks = seq(-100, 100, by=10)) + 
           geom_text(data = medians, 
                     aes(label = sprintf("%1.1f", medians$med), y = medians$med),
                     size = 4, vjust = -.3, fontface="bold",
                     position = position_dodge(.8))
}

one_time_medians_storm_geoMap <- function(dt, minn, maxx, ttl, subttl, differ=FALSE){
  # Storm related
  # one_time_medians means one-time in the sense of 25 years
  # strom, (not including 5, 10, 15, 20 years)

  tgt_col <- "twenty_five_years"
  x <- sapply(dt$location, 
              function(x) strsplit(x, "_")[[1]], 
              USE.NAMES=FALSE)
  lat <- as.numeric(x[1, ]); long <- as.numeric(x[2, ])
  dt$lat <- lat; dt$long <- long;

  states <- map_data("state")
  states_cluster <- subset(states, 
                           region %in% c("washington"))
  
  WA_counties <- map_data("county", "washington")
  WA_counties <- WA_counties %>% 
                 filter(subregion %in% c("whatcom", "skagit", "snohomish", "island"
                                         #, "okanogan", "chelan"
                                         ))%>% data.table()

  th <- theme(axis.title.y = element_blank(),
               axis.title.x = element_blank(),
               axis.ticks.y = element_blank(), 
               axis.ticks.x = element_blank(),
               axis.text = element_blank(),
               # plot.title=element_text(size = 14, face = "bold",
               #              margin = margin(t=.15, r=.1, b=-.5, l=0, "line")),
               plot.title = element_text(size=14, face = "bold"),
               legend.text = element_text(size=12, face="plain"),
               legend.title = element_blank(),
               legend.position = c(.93, .2),
               # legend.position = "top",
               strip.text = element_text(size=14, face="bold"))
  # color_limit <- max(abs(minn), abs(maxx))
  if (differ==TRUE){
    bbb <- dt %>%
           ggplot() +
           geom_polygon(data = WA_counties, 
                        aes(x=long, y=lat, group = group),
                        fill = "grey", color = "black") +
           geom_polygon(data=WA_counties, 
                        aes(x=long, y=lat,group = group), 
                        fill = NA, colour = "black", size=.5) + 
           geom_point(aes_string(x="long", y="lat",
                                 color = tgt_col), 
                      alpha=1, size=2.5) +
           scale_color_gradient2(midpoint = 0, mid = "white", 
                                 high = muted("blue"), low = muted("red"), 
                                 guide = "colourbar", space = "Lab",
                                 limits = c(minn, maxx)) +
          th +
          guides(fill = guide_colourbar(barwidth=.1, barheight=2),
                 direction = "vertical") + 
          ggtitle(ttl) #, subtitle=subttl
      } else{
       bbb <- dt %>%
              ggplot() +
              geom_polygon(data = WA_counties, 
                           aes(x = long, y = lat, group = group),
                           fill = "grey", color = "black") +
              geom_polygon(data=WA_counties, 
                           aes(x=long, y=lat,group = group), 
                          fill = NA, colour = "black", size=.5) + 
              geom_point(aes_string(x="long", y="lat", color = tgt_col), 
                         alpha = 1, size=2.5) +
              scale_color_viridis_c(option = "plasma", 
                                    name = "storm", direction = -1,
                                    limits = c(minn, maxx),
                                    breaks = pretty_breaks(n=8)) +
              th +
              guides(fill = guide_colourbar(barwidth=.1, barheight=2),
                     direction = "vertical") + 
              ggtitle(ttl)
    }
    return(bbb)
}

storm_box_plot <- function(data_tb){
  categ_lab <- sort(unique(data_tb$return_period))
  color_ord = c("grey47", "olivedrab4", "steelblue1", "gold") #  "red",
  x_ticks <- c("5", "10", "15", "20", "25")

  melted <- melt(data_tb,  id = c("location", "model", "cluster",
                                  "return_period", "emission"))
  
  ax_txt_size <- 5; ax_ttl_size <- 6; box_width = 0.53
  the <- theme(plot.margin = unit(c(t=.1, r=.2, b=.1, l=0.2), "cm"),
               panel.border = element_rect(fill=NA, size=.3),
               panel.grid.major = element_line(size = 0.05),
               panel.grid.minor = element_blank(),
               panel.spacing = unit(.35, "line"),
               legend.position = "bottom", 
               legend.key.size = unit(.6, "line"),
               legend.spacing.x = unit(.1, 'line'),
               panel.spacing.y = unit(.5, 'line'),
               legend.text = element_text(size = ax_ttl_size, face = "bold"),
               legend.margin = margin(t=.1, r=0, b=0, l=0, unit = 'line'),
               legend.title = element_blank(),
               plot.title = element_text(size = ax_ttl_size, face = "bold"),
               plot.subtitle = element_text(face = "bold"),
               strip.text.x = element_text(size = ax_ttl_size, face = "bold",
                                           margin = margin(.15, 0, .15, 0, "line")),
               strip.text.y = element_text(size = ax_ttl_size, face = "bold",
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
               axis.title.x = element_text(size = ax_ttl_size , face = "bold",
                                           margin = margin(t=2, r=0, b=-10, l=0))
                    )
  
  box_p <- ggplot(data = melted, 
                  aes(x=variable, y=value, fill=return_period)) +
           geom_hline(yintercept= 0, color = "red", size=.3) + 
           geom_boxplot(outlier.size = - 0.3, notch=F, 
                        width = box_width, lwd=.1, 
                        position = position_dodge(0.6), outlier.shape=NA) +
           facet_grid(~ emission ~ cluster, scales="free") + # , ncol=4 goes with facet_wrap
           scale_x_discrete(labels=c("five_years" = "5", 
                                     "ten_years" = "10",
                                     "fifteen_years" = "15",
                                     "twenty_years" = "20",
                                     "twenty_five_years" = "25")) + 
           xlab("time interval (years)") + 
           ylab("design storm intensity (mm/hr)") + 
           # ylim(quantile(melted$value, probs = c(0.05, 0.95))) + 
           scale_fill_manual(values = color_ord,
                             name = "Return\nPeriod", 
                             labels = categ_lab) + 
           scale_y_continuous(breaks = 1:20) + 
           the
}
#####################################
#####################################

geo_map_of_diffs <- function(dt_dt, col_col, minn, maxx, ttl, subttl){
  color_limit <- c(minn, maxx)
  x <- sapply(dt_dt$location, 
              function(x) strsplit(x, "_")[[1]], 
              USE.NAMES=FALSE)
  lat <- as.numeric(x[1, ]); long <- as.numeric(x[2, ])
  dt_dt$lat <- lat; dt_dt$long <- long;
  
  states <- map_data("state")
  states_cluster <- subset(states, 
                           region %in% c("washington"))
  WA_counties <- map_data("county", "washington")
  WA_counties <- WA_counties %>% 
                 filter(subregion %in% c("whatcom", "skagit", "snohomish", "island"
                                         #, "okanogan" "chelan"
                                         ))%>% data.table()

  the <- theme(axis.title.y = element_blank(),
               axis.title.x = element_blank(),
               axis.ticks.y = element_blank(), 
               axis.ticks.x = element_blank(),
               axis.text = element_blank(),
               plot.title=element_text(size = 14, face = "bold",
                                       margin = margin(t=.15, r=.1, b=0, l=0, "line")),
               legend.text = element_text(size = 8, face="plain"),
               legend.title = element_blank(),
               # legend.justification = c(.93, .9),
               legend.position=c(.93, .2),
               strip.text = element_text(size=14, face="bold"))

  dt_dt %>%
  ggplot() +
  geom_polygon(data=WA_counties, 
               aes(x=long, y=lat, group = group),
               fill = "grey", color = "black") +
  geom_polygon(data=WA_counties, 
               aes(x=long, y=lat, group = group), 
               fill = NA, colour = "black", size=.5) + 
  geom_point(aes_string(x="long", y="lat", color=col_col), 
             alpha = 1, size=2.5) +
  guides(fill = guide_colourbar(barwidt_dth=.1, barheight=2),
         direction = "vertical")+
  scale_color_viridis_c(option = "plasma", 
                        name = "storm", direction = -1,
                        limits = c(color_limit[1], color_limit[2]),
                        breaks = pretty_breaks(n=8)) +
  # scale_color_viridis_c(option = "plasma", 
  #                       name = "storm", direction = -1,
  #                       limits = c(min, max),
  #                       # begin = 0.5, end = 1,
  #                       breaks = pretty_breaks(n = 3)) +
  
  # scale_color_gradient2(breaks = c((as.integer(minn*0.6)), 
  #                                   0,
  #                                   (as.integer(maxx*0.9)), 
  #                                   (as.integer((maxx)*0.9))),
                        
  #                       labels = c((as.integer(minn*0.6)), 
  #                                  0, 
  #                                  (as.integer(maxx*0.9)),
  #                                  (as.integer((maxx)*0.9))),

  #                       low = "red", high = "blue", mid = "white",
  #                       space="Lab"
  #                       ) +

  # Look at this. This may work, if you spend time on it.
  # problem is that we have to create one of these
  # manually for each map? We do not want to do this.
  # There are too many maps, one function per map is just ...
  #
  # scale_fill_gradientn(name="CPU Utilization",
  #       colours=c("darkgreen","green","red","darkred","red","green","darkgreen"),
  #       values=c(0, 0.19, 0.2, 0.5, 0.8, 0.81, 1),
  #       limits=c(-color_limit, color_limit),
  #       breaks = c(20, 30, 40, 50, 60, 70, 80, 90, 100))
  # scale_color_gradient2(midpoint = 0, mid = "white", 
  #                       high=muted("blue"), low = muted("red"), 
  #                       guide="colourbar", space = "Lab",
  #                       limit=c(-color_limit, color_limit)) + 
  # scale_color_continuous(breaks = c(as.integer(minn+1), 0, as.integer(maxx-1)),
  #                        labels = c(as.integer(minn+1), 0, as.integer(maxx-1)),
  #                        low = "red", high = "blue") + 
  the +
  ggtitle(ttl) # , subtitle=subttl
}

all_mods_map_storm <- function(dt, minn, maxx, ttl){
  tgt_col <- "twenty_five_years"
  x <- sapply(dt$location, 
              function(x) strsplit(x, "_")[[1]], 
              USE.NAMES=FALSE)
  lat <- as.numeric(x[1, ]); long <- as.numeric(x[2, ])
  dt$lat <- lat; dt$long <- long;

  # states <- map_data("state")
  # states_cluster <- subset(states, 
  #                          region %in% c("washington"))

  WA_counties <- map_data("county", "washington")
  WA_counties <- WA_counties %>% 
                 filter(subregion %in% c("whatcom", "skagit", "snohomish", "island"
                                         #, "okanogan" "chelan"
                                         ))%>% data.table()
  dt %>%
  ggplot() +
  geom_polygon(data = WA_counties, 
               aes(x = long, y = lat, group = group),
               fill = "grey", color = "black") +
  geom_polygon(data=WA_counties, 
               aes(x=long, y=lat,group = group), 
               fill=NA, colour = "black", size=.5) + 
  geom_point(aes_string(x="long", y="lat", color = tgt_col), 
             alpha=1, size=2.5) +
  facet_wrap(~ model) +
  scale_color_viridis_c(option = "plasma", 
                        name = "storm", direction = -1,
                        limits = c(minn, maxx),
                        breaks = pretty_breaks(n = 4)) +
  theme(axis.title.y = element_blank(),
        axis.title.x = element_blank(),
        axis.ticks.y = element_blank(), 
        axis.ticks.x = element_blank(),
        axis.text = element_blank(),
        plot.title = element_text(size = 14, face = "bold"),
        legend.text = element_text(size = 12, face="plain"),
        legend.title = element_blank(),
        legend.position = "top",
        legend.margin = margin(t=-1, r=0, b=0, l=0, unit = 'line'),
        strip.text = element_text(size=14, face="bold")) + 
  ggtitle(ttl)
}

obs_hist_map_storm <- function(dt, minn, maxx, fips_clust, tgt_col="twenty_five_years") {
  dt <- add_coord_from_location(dt)
  dt <- merge(dt, fips_clust, by="location", all.x=T)
  dt <- suppressWarnings({within(dt, 
                          remove(location, model, return_period))})
  # states <- map_data("state")
  # states_cluster <- subset(states, 
  #                          region %in% c("washington"))
  WA_counties <- map_data("county", "washington")
  WA_counties <- WA_counties %>% 
                 filter(subregion %in% c("whatcom", "skagit", "snohomish", "island"
                                         # , "okanogan", "chelan"
                                         ))%>% data.table()
  dt %>%
  ggplot() +
  geom_polygon(data = WA_counties, 
               aes(x = long, y = lat, group = group),
               fill = "grey", color = "black") +
  geom_polygon(data=WA_counties, 
               aes(x=long, y=lat,group = group), 
               fill = NA, colour = "black", size=.5) + 
  geom_point(aes_string(x = "long", y = "lat",
                        color = tgt_col), alpha = 1, size=2.5) +
  scale_color_viridis_c(option = "plasma", 
                        name = "storm", direction = -1,
                        limits = c(minn, maxx),
                        breaks = pretty_breaks(n=8)) +
  theme(axis.title.y = element_blank(),
        axis.title.x = element_blank(),
        axis.ticks.y = element_blank(), 
        axis.ticks.x = element_blank(),
        axis.text = element_blank(),
        plot.title = element_text(size = 14, face = "bold"),
        legend.text = element_text(size = 12, face="plain"),
        legend.title = element_blank(),
        # legend.justification = c(.93, .2),
        legend.position=c(.93, .2),
        strip.text = element_text(size=14, face="bold"))
  # +
  # geom_polygon(fill = "transparent", color = "red", size = .1, 
  #               data = dt, aes(x = long, y = lat, group = cluster))
}
############################################################
#      
#            Precip
#
#
ann_wtrYr_chunk_cumP_box_cluster_x <- function(dt, y_lab, tgt_col){
  # toss unwanted time periods
  dt <- dt %>% 
        filter(time_period != "1950-2005" & 
               time_period != "2006-2025") %>% 
        data.table()

  dt <- within(dt, remove(month, day, precip, model, wtr_yr))
  # dt <- cluster_numeric_2_str(dt)
  # medians <- data.frame(dt) %>% 
  #            group_by(cluster, time_period, emission) %>% 
  #            summarise( medians = median(get(tgt_col)))  %>% 
  #            data.table()

  melted <- melt(dt, id = c("location", "year", "time_period", "emission", "cluster"))
  
  categ_label <-  c("Western coastal", "Cascade foothills", 
                   "Northwest Cascades", "Northcentral Cascades", 
                   "Northeast Cascades")
  time_label <- c("1979-2016", "2026-2050", "2051-2075", "2076-2099")
  melted$cluster <- factor(melted$cluster, levels=categ_label)
  melted$time_period <- factor(melted$time_period, levels=time_label)
  
  color_ord = c("grey47", "dodgerblue2", "olivedrab4", "gold")
  ax_txt_size <- 6; ax_ttl_size <- 8; box_width = 0.53

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
               axis.title.x = element_text(size = ax_ttl_size , face = "bold",
                                           margin = margin(t=2, r=0, b=-10, l=0))
                    )

  box_p <- ggplot(data = melted, 
                  aes(x=cluster, y=value, fill=time_period)) +
           the + 
           geom_hline(yintercept= 0, color = "red", size=.3) +
           geom_boxplot(outlier.size = - 0.3, notch=F, 
                        width = box_width, lwd=.1, 
                        position = position_dodge(0.6), outlier.shape=NA) +
           facet_grid(~ emission, scales="free") +
           xlab("precip. group") +
           ylab(y_lab) + 
           scale_fill_manual(values = color_ord, labels = time_label)
  return(box_p)
}

cum_box_cluster_x <- function(dt, tgt_col, y_lab){
  dt <- dt %>% 
        filter(time_period != "1950-2005" & time_period != "2006-2025") %>% 
        data.table()

  time_label <- c(# "1950-2005", 
                  "1979-2016", # "2006-2025",
                  "2026-2050",
                  "2051-2075", "2076-2099")

  color_ord = c("grey47", "dodgerblue2", "olivedrab4", "gold")
  # color_ord = c("grey47", "dodgerblue2", "olivedrab4", "red", "blue3", "gold")
  
  # medians <- data.frame(dt) %>% 
  #            group_by(cluster, time_period, emission) %>% 
  #            summarise( medians = median(get(tgt_col)))  %>% 
  #            data.table()
  dt <- within(dt, remove( day, precip, wtr_yr))
  if (tgt_col=="annual_cum_precip"){
    dt <- within(dt, remove(month, day))
  }  
  melted <- melt(dt, id = c("location", "year", 
                            "time_period", "model", "emission",
                            "cluster"))
  
  melted$cluster <- factor(melted$cluster, levels=c(4, 3, 2, 1))

  ax_txt_size <- 6; ax_ttl_size <- 7; box_width = 0.53
  the <- theme(plot.margin = unit(c(t=.1, r=.2, b=.1, l=0.2), "cm"),
               panel.border = element_rect(fill=NA, size=.3),
               panel.grid.major = element_line(size = 0.05),
               panel.grid.minor = element_blank(),
               panel.spacing = unit(.35, "line"),
               legend.position = "bottom", 
               legend.key.size = unit(.6, "line"),
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
               axis.title.x = element_text(size = ax_ttl_size , face = "bold",
                                           margin = margin(t=2, r=0, b=-10, l=0))
                    )

  box_p <- ggplot(data = melted, 
                  aes(x=cluster, y=value, fill=time_period)) +
           the + 
           geom_hline(yintercept= 0, color = "red", size=.3) + 
           geom_boxplot(outlier.size = - 0.3, notch=F, 
                        width = box_width, lwd=.1, 
                        position = position_dodge(0.6), outlier.shape=NA) +
           facet_grid(~ emission, scales="free") +
           ylab(y_lab) + 
           scale_fill_manual(values = color_ord, labels = time_label)
           
  
  return(box_p)
}

cum_clust_box_plots <- function(dt, tgt_col, y_lab){
  # color_ord = c("red", "purple", "dodgerblue2", "blue4")
  color_ord = c("grey47", "dodgerblue2", "olivedrab4", "red")
  
  # medians <- data.frame(dt) %>% 
  #            group_by(cluster, time_period, emission) %>% 
  #            summarise( medians = median(get(tgt_col)))  %>% 
  #            data.table()

  dt <- within(dt, remove(day, precip, wtr_yr))
  if (tgt_col=="annual_cum_precip"){
    dt <- within(dt, remove(month, day))
  }

  # dt <- dt %>% filter(get(tgt_col) >= 0)%>% data.table()
  
  melted <- melt(dt, id = c("location", "year", 
                            "time_period", "model", "emission",
                            "cluster"))
  
  melted$cluster <- factor(melted$cluster, levels=c(4, 3, 2, 1))

  ax_txt_size <- 6; ax_ttl_size <- 7; box_width = 0.53
  the <- theme(plot.margin = unit(c(t=.1, r=.2, b=.1, l=0.2), "cm"),
               panel.border = element_rect(fill=NA, size=.3),
               panel.grid.major = element_line(size = 0.05),
               panel.grid.minor = element_blank(),
               panel.spacing = unit(.35, "line"),
               legend.position = "bottom", 
               legend.key.size = unit(.6, "line"),
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
               axis.title.x = element_text(size = ax_ttl_size , face = "bold",
                                           margin = margin(t=2, r=0, b=-10, l=0))
                    )

  box_p <- ggplot(data = melted, 
                  aes(x=time_period, y=value, fill=cluster)) +
           the +
           geom_hline(yintercept= 0, color = "red", size=.3) + 
           geom_boxplot(outlier.size = - 0.3, notch=F, 
                        width = box_width, lwd=.1, 
                        position = position_dodge(0.6), outlier.shape=NA) +
           facet_grid(~ emission, scales="free") +
           xlab("time period") + 
           ylab(y_lab) + 
           scale_fill_manual(values = color_ord, name = "precip\nlevel")
           
  return(box_p)
}

geo_map_of_clusters <- function(obs_w_clusters){
  obs_w_clusters <- subset(obs_w_clusters, 
                           select=c(location, cluster))
  obs_w_clusters <- unique(obs_w_clusters)

  x <- sapply(obs_w_clusters$location, 
              function(x) strsplit(x, "_")[[1]], 
              USE.NAMES=FALSE)
  lat = as.numeric(x[1, ]); long = as.numeric(x[2, ])

  obs_w_clusters$lat <- lat
  obs_w_clusters$long <- long
  obs_w_clusters <- within(obs_w_clusters, remove(location))

  # states <- map_data("state")
  # WA_state <- subset(states, region %in% c("washington"))
  # WA_state <- WA_state %>% filter(subregion == "main")
  
  WA_counties <- map_data("county", "washington")
  WA_counties <- WA_counties %>% 
                 filter(subregion %in% c("whatcom", "skagit", "snohomish", "island"
                                         #, "okanogan" "chelan"
                                         ))%>% 
                 data.table()

  # color_ord = c("red", "purple", "dodgerblue2", "blue4")
  if (length(unique(obs_w_clusters$cluster))==4){
     color_ord = c("royalblue3", "steelblue1", "maroon3", "red")
     } else if(length(unique(obs_w_clusters$cluster))== 5){
     color_ord = c("royalblue3", "steelblue1", "maroon3", "red", "black")
  }
  
  the_theme <- theme(plot.margin = unit(c(t=.2, r=.2, b=.2, l=0.2), "cm"),
                     panel.border = element_rect(fill=NA, size=.3),
                     legend.position = "bottom",
                     legend.key.size = unit(.6, "line"),
                     legend.spacing.x = unit(.1, 'line'),
                     panel.spacing.y = unit(.5, 'line'),
                     legend.text = element_text(size = 9, face="bold"),
                     legend.margin = margin(t=.4, r=0, b=0, l=0, unit = 'line'),
                     legend.title = element_blank(),
                     plot.title = element_text(size = 13, face = "bold"),
                     plot.subtitle = element_text(size = 10, face = "bold"),
                     axis.ticks = element_blank(),
                     axis.text.y = element_blank(),
                     axis.text.x = element_blank(),
                     axis.title.y = element_blank(),
                     axis.title.x = element_blank())

  cluster_plot <- obs_w_clusters %>%
                  ggplot() +
                  geom_polygon(data = WA_counties, 
                               aes(x=long, y=lat, group = group),
                               fill = "grey", color = "black", size=0.5) +
                  geom_polygon(data=WA_counties, 
                               aes(x=long, y=lat,group = group), 
                               fill = NA, colour = "black", size=0.0000001) + 
                  geom_point(aes_string(x = "long", y = "lat", color="cluster"), 
                            alpha = 1, size=2.5) + 
                  scale_color_manual(values = color_ord, name = "Precip.") + 
                  the_theme +
                  # size of dot inside the legend
                  guides(colour = guide_legend(override.aes = list(size=3))) + 
                  # labs(title = "Groups of grids based on annual precip.",
                  #      subtitle = "averaged over 38 years.") + 
                  # ggtitle("Groups of grids") + 
                  # size of dot inside the legend box
                  guides(colour = guide_legend(override.aes = list(size=2.5)))

  return(cluster_plot)
}

satellite_map_of_clusters <- function(obs_w_clusters){
  obs_w_clusters <- subset(obs_w_clusters, 
                           select=c(location, cluster))
  obs_w_clusters <- unique(obs_w_clusters)
  
  x <- sapply(obs_w_clusters$location, 
              function(x) strsplit(x, "_")[[1]], 
              USE.NAMES=FALSE)
  lat = as.numeric(x[1, ]); long = as.numeric(x[2, ])
  
  obs_w_clusters$lat <- lat
  obs_w_clusters$long <- long
  obs_w_clusters <- within(obs_w_clusters, remove(location))
       
  sh_dir_1 <- "/Users/hn/Documents/GitHub/large_4_GitHub/"
  sh_dir <- paste0(sh_dir_1, "4_analog_web_site/tl_2017_us_county_simple/")
  counties <- rgdal::readOGR(dsn=path.expand(sh_dir), 
                             layer = "tl_2017_us_county")

  # Extract just the three states OR: 41, WA:53, ID: 16
  counties <- counties[counties@data$STATEFP %in% c("53"), ]

  # ggplot(counties) + 
  # aes(long, lat, group=group) + 
  # geom_polygon() +
  # geom_path(color="white") +
  # coord_equal()

  # ggplot()+ 
  # geom_polygon(data=counties, aes(long, lat, group = group, fill = hole), 
  #              colour = alpha("white", 1/2), size = 0.7) + 
  # scale_fill_manual(values = c("grey47", "white")) + 
  # theme(legend.position="none")

  # "grey47",
  color_ord = c("blue4", "dodgerblue2", "purple", "red") 
  categ_lab =  c("Western coastal", "Cascade foothills", 
                   "Northwest Cascades", "Northcentral Cascades", 
                   "Northeast Cascades")

  the_theme <- theme(plot.margin = unit(c(t=.2, r=.2, b=.2, l=0.2), "cm"),
                     panel.border = element_rect(fill=NA, size=.3),
                     legend.position = "bottom",
                     legend.key.size = unit(1, "line"),
                     legend.spacing.x = unit(.1, 'line'),
                     panel.spacing.y = unit(.5, 'line'),
                     legend.text = element_text(size = 10, face="bold"),
                     legend.margin = margin(t=.4, r=0, b=0, l=0, unit = 'line'),
                     legend.title = element_blank(),
                     plot.title = element_text(size = 13, face = "bold"),
                     plot.subtitle = element_text(size = 9, face = "bold"),
                     axis.ticks = element_blank(),
                     axis.text.y = element_blank(),
                     axis.text.x = element_blank(),
                     axis.title.y = element_blank(),
                     axis.title.x = element_blank())
   
  cluster_plot <- obs_w_clusters %>%
                  ggplot() +
                  qmap('washington', zoom = 6, maptype = 'satellite') +
                  geom_polygon(data = WA_state, 
                              aes(x=long, y=lat, group = group),
                                  fill = "grey", color = "black", size=0.5) +
                  geom_point(aes_string(x = "long", y = "lat", color="cluster"), 
                             alpha = 1, size=2.5) + 
                  scale_color_manual(values = color_ord, name = "Precip.\n", 
                                     labels = categ_lab) + 
                  the_theme +
                  # size of dot inside the legend
                  guides(colour = guide_legend(override.aes = list(size=3))) + 
                  labs(title = "Groups of grids based on annual precip.",
                       subtitle = "averaged over 38 years.")
                  ggtitle("Groups of grids")
  return(cluster_plot)
}
##########################################################################
#####                                                         ############
#####    after meeting with Nichole and changing clusters     ############
#####                                                         ############
#####                                                         ############
##########################################################################
geo_map_of_precip <- function(data_dt){
  data_dt <- unique(data_dt)

  x <- sapply(data_dt$location, 
              function(x) strsplit(x, "_")[[1]], 
              USE.NAMES=FALSE)
  lat = as.numeric(x[1, ]); long = as.numeric(x[2, ])

  data_dt$lat <- lat
  data_dt$long <- long
  data_dt <- within(data_dt, remove(location))

  WA_counties <- map_data("county", "washington")
  WA_counties <- WA_counties %>% 
                 filter(subregion %in% c("whatcom", "skagit", "snohomish", "island"
                                         #, "okanogan" "chelan"
                                         ))%>% 
                 data.table()

  the_theme <- theme(plot.margin = unit(c(t=.2, r=.2, b=.2, l=0.2), "cm"),
                     panel.border = element_rect(fill=NA, size=.3),
                     # legend.position = "right",
                     legend.position = c(0.8, 0.2),
                     legend.key.size = unit(.6, "line"),
                     legend.spacing.x = unit(.1, 'line'),
                     panel.spacing.y = unit(.5, 'line'),
                     legend.text = element_text(size = 9, face="bold"),
                     legend.margin = margin(t=.4, r=0, b=0, l=0, unit = 'line'),
                     legend.title = element_blank(),
                     plot.title = element_text(size = 13, face = "bold"),
                     plot.subtitle = element_text(size = 10, face = "bold"),
                     axis.ticks = element_blank(),
                     axis.text.y = element_blank(),
                     axis.text.x = element_blank(),
                     axis.title.y = element_blank(),
                     axis.title.x = element_blank())

  data_dt %>%
  ggplot() +
  geom_polygon(data = WA_counties, 
               aes(x=long, y=lat, group = group),
               fill = "grey", color = "black", size=0.5) +
  geom_polygon(data=WA_counties, 
               aes(x=long, y=lat, group = group), 
               fill = NA, colour = "black", size=0.0000001) + 
  geom_point(aes_string(x = "long", y = "lat", color="ann_prec_mean"), 
            alpha = 1, size=2.5) + 
  the_theme +
  scale_color_viridis_c(option = "plasma", 
                        name = "storm", direction = -1,
                        limits = c(500, 5000),
                        breaks = pretty_breaks(n = 4)) + 
  labs(title = "avg. annual precip.") # subtitle = "averaged over 38 years."
}

geo_map_of_elevation <- function(data_dt){
  x <- sapply(data_dt$location, 
              function(x) strsplit(x, "_")[[1]], 
              USE.NAMES=FALSE)
  lat = as.numeric(x[1, ]); long = as.numeric(x[2, ])

  data_dt$lat <- lat; data_dt$long <- long
  data_dt <- within(data_dt, remove(location))

  data_dt <- unique(data_dt)
  WA_counties <- map_data("county", "washington")
  WA_counties <- WA_counties %>% 
                 filter(subregion %in% c("whatcom", "skagit", "snohomish",
                                         "island"
                                         #, "okanogan", "chelan"
                                         )) %>% data.table()

  the_theme <- theme(plot.margin = unit(c(t=.2, r=.2, b=.2, l=0.2), "cm"),
                     panel.border = element_rect(fill=NA, size=.3),
                     # legend.position = "right",
                     legend.position = c(0.8, 0.2),
                     legend.key.size = unit(.6, "line"),
                     legend.spacing.x = unit(.1, 'line'),
                     panel.spacing.y = unit(.5, 'line'),
                     legend.text = element_text(size = 9, face="bold"),
                     legend.margin = margin(t=.4, r=0, b=0, l=0, unit = 'line'),
                     legend.title = element_blank(),
                     plot.title = element_text(size = 13, face = "bold"),
                     plot.subtitle = element_text(size = 10, face = "bold"),
                     axis.ticks = element_blank(),
                     axis.text.y = element_blank(),
                     axis.text.x = element_blank(),
                     axis.title.y = element_blank(),
                     axis.title.x = element_blank())

  data_dt %>%
  ggplot() +
  geom_polygon(data = WA_counties, 
               aes(x=long, y=lat, group = group),
               fill = "grey", color = "black", size=0.5) +
  geom_polygon(data=WA_counties, 
               aes(x=long, y=lat, group = group), 
               fill = NA, colour = "black", size=0.0000001) + 
  geom_point(aes_string(x = "long", y = "lat", color="elevation"), 
            alpha = 1, size=2.5) + 
  the_theme +
  scale_color_viridis_c(option = "plasma", 
                        name = "storm", direction = -1,
                        limits = c(-100, 4000),
                        breaks = pretty_breaks(n = 4)) + 
  labs(title = "map of elevations")
}

geo_map_of_rain_frac <- function(data_dt){
  data_dt <- unique(data_dt)

  x <- sapply(data_dt$location, 
              function(x) strsplit(x, "_")[[1]], 
              USE.NAMES=FALSE)
  lat = as.numeric(x[1, ]); long = as.numeric(x[2, ])

  data_dt$lat <- lat
  data_dt$long <- long
  data_dt <- within(data_dt, remove(location))

  WA_counties <- map_data("county", "washington")
  WA_counties <- WA_counties %>% 
                 filter(subregion %in% c("whatcom", "skagit", "snohomish",
                                         "island"
                                         # , "okanogan", "chelan", 
                                         ))%>% 
                 data.table()

  the_theme <- theme(plot.margin = unit(c(t=.2, r=.2, b=.2, l=0.2), "cm"),
                     panel.border = element_rect(fill=NA, size=.3),
                     # legend.position = "right",
                     legend.position = c(0.8, 0.2),
                     legend.key.size = unit(.6, "line"),
                     legend.spacing.x = unit(.1, 'line'),
                     panel.spacing.y = unit(.5, 'line'),
                     legend.text = element_text(size = 9, face="bold"),
                     legend.margin = margin(t=.4, r=0, b=0, l=0, unit = 'line'),
                     legend.title = element_blank(),
                     plot.title = element_text(size = 13, face = "bold"),
                     plot.subtitle = element_text(size = 10, face = "bold"),
                     axis.ticks = element_blank(),
                     axis.text.y = element_blank(),
                     axis.text.x = element_blank(),
                     axis.title.y = element_blank(),
                     axis.title.x = element_blank())
  data_dt$avg_rain_frac <- data_dt$avg_rain_frac*100

  data_dt %>%
  ggplot() +
  geom_polygon(data = WA_counties, 
               aes(x=long, y=lat, group = group),
               fill = "grey", color = "black", size=0.5) +
  geom_polygon(data=WA_counties, 
               aes(x=long, y=lat, group = group), 
               fill = NA, colour = "black", size=.5) + 
  geom_point(aes_string(x = "long", y = "lat", color="avg_rain_frac"), 
            alpha = 1, size=2.5) + 
  the_theme +
  scale_color_viridis_c(option = "plasma", 
                        name = "storm", direction = -1,
                        limits = c(0, 100),
                        breaks = pretty_breaks(n = 4)) + 
  labs(title = paste0("avg. rain fraction (", unique(data_dt$time_period), ")"))
}

geo_map_of_diffs_discrete_cuts <- function(dt, col_col, ttl, subttl){
  disc_colors <- c("white", "lightskyblue","deepskyblue",
                   "dodgeblue3", "dodgerblue4", "blue4")
  x <- sapply(dt$location,
              function(x) strsplit(x, "_")[[1]], 
              USE.NAMES=FALSE)

  lat <- as.numeric(x[1, ]); long <- as.numeric(x[2, ])
  dt$lat <- lat; dt$long <- long;
  
  states <- map_data("state")
  states_cluster <- subset(states, 
                           region %in% c("washington"))
  WA_counties <- map_data("county", "washington")
  WA_counties <- WA_counties %>% 
                 filter(subregion %in% c("whatcom", "skagit", "snohomish", "island"
                                         # , "okanogan", "chelan"
                                         ))%>% 
                 data.table() 
  
  dt %>%
  ggplot() +
  geom_polygon(data = WA_counties, 
               aes(x = long, y = lat, group = group),
               fill = "grey", color = "black") +
  geom_polygon(data=WA_counties, 
               aes(x=long, y=lat, group = group), 
               fill = NA, colour = "black", size=.5) + 
  geom_point(aes_string(x = "long", y = "lat", color = col_col), 
             alpha=1, size=2.5) +
  scale_color_manual(name = "qsec",
                     values = c("(-Inf,-0.5]"= "red4",
                                "(-0.5,-0.2]" = "red",
                                "(-0.2,-0.15]" = "brown1",
                                "(-0.15,-0.1]" = "tomato1",
                                "(-0.1,-0.05]" = "hotpink1",
                                "(-0.05,0]" = "white",
                                "(0,0.05]"  = "white",
                                "(0.05,0.1]" = "yellow",
                                "(0.1,0.15]" = "deepskyblue",
                                "(0.15,0.2]" = "dodgerblue3",
                                "(0.2,0.5]"  = "dodgerblue4",
                                "(0.5, Inf]"  = "blue4"
                                ),
                     # labels = c("<= -50%", 
                     #            "-50% < . <= -20%",
                     #            "-20% < . <= -15%",
                     #            "-15% < . <= -10%",
                     #            "-10% < . <= -5%",
                     #            "-5% < . <= 0%",
                     #            "0% < . <= 5%",
                     #            "5% < . <= 10%",
                     #            "10% < . <= 15%",
                     #            "15% < . <= 20%",
                     #            "20% < . <= 50%",
                     #            ">= 50%"
                     #            )
                     )+
  theme(axis.title.y = element_blank(),
        axis.title.x = element_blank(),
        axis.ticks.y = element_blank(), 
        axis.ticks.x = element_blank(),
        axis.text = element_blank(),
        plot.title = element_text(size = 14, face = "bold"),
        legend.text = element_text(size = 10, face="plain"),
        legend.title = element_blank(),
        legend.position = "bottom",
        strip.text = element_text(size=14, face="bold")) +  
  guides(colour = guide_legend(override.aes = list(size=2))) +
  ggtitle(ttl, subtitle=subttl)
  
}

box_dt_25_clust_x <- function(dt_25){
  categ_lab <- sort(unique(dt_25$return_period))
  
  if (length(unique(dt_25$return_period)) == 3){
    color_ord = c("dodgerblue2", "olivedrab4", "gold")
    } else if (length(unique(dt_25$return_period)) == 4){
      color_ord = c("grey47", "dodgerblue2", "olivedrab4", "gold")
    } else if (length(unique(dt_25$return_period)) == 5){
    color_ord = c("red", "grey47", "dodgerblue2", "olivedrab4", "gold")
  }

  medians <- data.frame(dt_25) %>% 
             group_by(return_period, emission, cluster) %>% 
             summarise(med_25 = median(twenty_five_years)) %>% 
             data.table()

  melted <- melt(dt_25, 
                 id = c("cluster", "return_period", "emission"))

  ax_txt_size <- 8; ax_ttl_size <- 10; box_width = 0.65
  ax_txt_size <- 6; ax_ttl_size <- 8; box_width = 0.6
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
               legend.margin = margin(t=0, r=0, b=0, l=0, unit = 'line'),
               legend.title = element_blank(),
               plot.title = element_text(size = ax_ttl_size, face = "bold",
                                         margin = margin(t=.15, r=.1, b=0, l=0, "line")),
               plot.subtitle = element_text(size=ax_txt_size, face = "plain"),
               strip.text.x = element_text(size = ax_ttl_size, face = "bold",
                                           margin = margin(.15, 0, .15, 0, "line")),
               axis.ticks = element_line(size = .1, color = "black"),
               axis.text.y = element_text(size = ax_txt_size, face = "bold", 
                                          color = "black"),
               axis.text.x = element_text(size = ax_txt_size, face = "bold", 
                                          color="black",
                                          margin=margin(t=.05, r=5, l=5, b=0,"pt")
                                          ),
               axis.title.y = element_text(size = ax_ttl_size, face = "bold", 
                                           margin = margin(t=0, r=2, b=0, l=0)),
               axis.title.x = element_blank()
              )

  box_p <- ggplot(data = melted, 
                  aes(x=cluster, y=value, fill=return_period)) +
           # geom_hline(yintercept= 0, color = "red", size=.3) +
           geom_boxplot(outlier.size = -0.3, notch=F, 
                        width = box_width, lwd=.1, 
                        position = position_dodge(0.85), 
                        outlier.shape=NA
                        ) +
           the +
           # facet_grid(~ cluster) +
           # xlab("precip. group") + 
           ylab("design storm intensity (mm/hr)") + 
           # ylim(quantile(melted$value, probs = c(0.05, 0.95))) + 
           scale_fill_manual(values = color_ord,
                             name = "Return\nPeriod", 
                             labels = categ_lab) + 
           scale_y_continuous(breaks = seq(0, 20, by=2)) + 
           geom_text(data = medians, 
                     aes(label = sprintf("%1.1f", medians$med_25), 
                         y = medians$med_25),
                     size = 2, fontface = "bold",
                     position = position_dodge(.8), vjust = -.3)
 return(box_p)
}

storm_diff_box_25yr_clust_x <- function(data_tb, tgt_col){
  data_tb <- data_tb %>% 
             filter(time_interval=="twenty_five_years") %>% 
             data.table()

  needed_cols <- c("return_period", "emission", "cluster", tgt_col)
  data_tb <- subset(data_tb, select=needed_cols)

  time_label <- sort(unique(data_tb$return_period))
  data_tb$return_period <- factor(data_tb$return_period, levels=time_label)
  data_tb$cluster <- factor(data_tb$cluster, 
                            levels=c("Western coastal", 
                                     "Cascade foothills", 
                                     "Northwest Cascades", 
                                     "Northcentral Cascades", 
                                     "Northeast Cascades"))
     
  if (length(time_label) == 3){
    color_ord = c("dodgerblue2", "olivedrab4", "gold")
    } else if (length(time_label) == 4){
      color_ord = c("grey47", "dodgerblue2", "olivedrab4", "gold")
    } else if (length(time_label) == 5){
    color_ord = c("red", "grey47", "dodgerblue2", "olivedrab4", "gold")
  }
  ax_txt_size <- 8; ax_ttl_size <- 10; box_width = 0.53
  ax_txt_size <- 6; ax_ttl_size <- 8; box_width = 0.6
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
               legend.margin = margin(t=0, r=0, b=0, l=0, unit = 'line'),
               legend.title = element_blank(),
               plot.title = element_text(size = ax_ttl_size, face = "bold",
                                         margin = margin(t=.15, r=.1, b=0, l=0, "line")),
               plot.subtitle = element_text(size=ax_txt_size, face = "plain"),
               strip.text.x = element_text(size = ax_ttl_size, face = "bold",
                                           margin = margin(.15, 0, .15, 0, "line")),
               axis.ticks = element_line(size = .1, color = "black"),
               axis.text.y = element_text(size = ax_txt_size, face = "bold", 
                                          color = "black"),
               axis.text.x = element_text(size = ax_txt_size, face = "bold", 
                                          color="black",
                                          margin=margin(t=.05, r=5, l=5, b=0,"pt")
                                          ),
               axis.title.y = element_text(size = ax_ttl_size, face = "bold", 
                                           margin = margin(t=0, r=2, b=0, l=0)),
               axis.title.x = element_blank()
              )

  if (tgt_col=="perc_diff"){
     y_labb <- "differences (%)"
     } else {
      y_labb <- "magnitude of differences"
  }
  # box_title <- "diff. of 25 yr/24 hr. design storm"

  medians <- data.frame(data_tb) %>% 
             group_by(return_period, emission, cluster) %>% 
             summarise( med = median(get(tgt_col))) %>% 
             data.table()

  box_p <- ggplot(data = data_tb, 
                  aes(x=cluster, y=get(tgt_col), fill=return_period)) +
           geom_hline(yintercept= 0, color = "red", size=.3) + 
           geom_boxplot(outlier.size = - 0.3, notch=F, 
                        width = box_width, lwd=.1, 
                        position = position_dodge(0.8), outlier.shape=NA
                        ) +
           # labs(x="", y="") + # theme_bw() + 
           # facet_grid(~ emission, scales="free") + # , ncol=4 goes with facet_wrap
           ylab(y_labb) + 
           scale_fill_manual(values = color_ord,
                             name = "Return\nPeriod", 
                             labels = time_label) + 
           the +
           scale_y_continuous(breaks = seq(-100, 100, by=10)) + 
           geom_text(data = medians, 
                     aes(label = sprintf("%1.1f", medians$med), y = medians$med),
                     size = 2, fontface = "bold",
                     position = position_dodge(.8), vjust = -.3)
}

##############################################################
########
########           Jan 6th, 2020
########           Separate clusters for annual plots
########           to make them similar to monthly and seasonal.
########           This sucks
########
##############################################################

ann_box_sep_cluster <- function(dt, y_lab, tgt_col, ttl, subttl){

  # toss unwanted time periods
  dt <- dt %>% 
        filter(# time_period != "1950-2005" & 
               time_period != "2006-2025") %>% 
        data.table()

  dt <- subset(dt, select=c("time_period", "emission", # "location",
                            "cluster", tgt_col))
  medians <- data.frame(dt) %>% 
             group_by(cluster, time_period, emission) %>% 
             summarise(med = median(get(tgt_col))) %>% 
             data.table()
  melted <- melt(dt, id = c("emission", # "location", 
                            "time_period", "cluster")); rm(dt)
  time_label <- sort(unique(melted$time_period))
  if (length(unique(melted$time_period)) == 3){
     color_ord = c("dodgerblue2", "olivedrab4", "gold")
     } else if (length(unique(melted$time_period)) == 4){
       color_ord = c("grey47", "dodgerblue2", "olivedrab4", "gold")
     } else if (length(unique(melted$time_period)) == 5){
     color_ord = c("red", "grey47", "dodgerblue2", "olivedrab4", "gold")
  }

  categ_label <- c("Western coastal", "Cascade foothills", 
                   "Northwest Cascades", "Northcentral Cascades", 
                   "Northeast Cascades")
  melted$cluster <- factor(melted$cluster, levels=categ_label)
  melted$time_period <- factor(melted$time_period, levels=time_label)

  ax_txt_size <- 6; ax_ttl_size <- 8; box_width = 0.4
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
               legend.margin = margin(t=0, r=0, b=0, l=0, unit = 'line'),
               legend.title = element_blank(),
               plot.title = element_text(size = ax_ttl_size, face = "bold",
                                         margin = margin(t=.15, r=.1, b=0, l=0, "line")),
               plot.subtitle = element_text(size=ax_txt_size, face = "plain"),
               strip.text.x = element_text(size = ax_ttl_size, face = "bold",
                                           margin = margin(.15, 0, .15, 0, "line")),
               axis.ticks.y = element_line(size = .1, color = "black"),
               axis.ticks.x = element_blank(),
               axis.text.y = element_text(size = ax_txt_size, face = "bold", 
                                          color = "black"),
               axis.text.x = element_blank(),
               axis.title.y = element_text(size = ax_ttl_size, face = "bold", 
                                           margin = margin(t=0, r=2, b=0, l=0)),
               axis.title.x = element_blank()
              )
  signif <- if (grepl("diff", tgt_col)) "%1.1f" else "%1.0f"
  ########
  ########    PLOT
  ########
  bx <- ggplot(data = melted, aes(x=cluster, y=value, fill=time_period)) +
        the + 
        # geom_hline(yintercept= 0, color = "red", size=.3) +
        geom_boxplot(outlier.size = -0.3, notch=F, 
                     width = box_width, lwd=.1,
                     position = position_dodge(0.6), outlier.shape=NA) +
        scale_x_discrete(expand=c(0.1, 0)) + 
        ylab(y_lab) + 
        scale_fill_manual(values = color_ord, labels = time_label) +
        geom_text(data = medians, 
                  aes(label = sprintf(signif, medians$med), y = medians$med), 
                      size = 2, fontface = "bold",
                      position = position_dodge(.6), vjust = -.3)

  if (tgt_col=="perc_diff"){
   bx <- bx + geom_hline(yintercept= 0, color = "red", size=.3)
  }
   return(bx)  
}

annual_frac_sep_clust <-function(data_tb,y_lab="rain fraction (%)",tgt_col="rain_fraction"){
  data_tb$rain_fraction <- data_tb$rain_fraction * 100
  data_tb$snow_fraction <- data_tb$snow_fraction * 100
  if (tgt_col=="rain_fraction"){
     data_tb <- within(data_tb, remove(model, year, location, 
                                       annual_cum_precip, snow_fraction))
     } else {
       data_tb <- within(data_tb, remove(model, year, location, 
                                         annual_cum_precip, rain_fraction))
   }

  region_levels <- c("Western coastal", "Cascade foothills", 
                     "Northwest Cascades", "Northcentral Cascades", 
                     "Northeast Cascades")
  data_tb$cluster <- factor(data_tb$cluster, levels=region_levels, order=T)
  medians <- data.frame(data_tb) %>% 
             group_by(cluster, time_period, emission) %>% 
             summarise(med = median(get(tgt_col))) %>% 
             data.table()

  melted <- melt(data_tb, id = c("emission", "time_period", "cluster"))
  rm(data_tb)
  
  time_label <- sort(unique(melted$time_period))
  if (length(unique(melted$time_period)) == 3){
     color_ord = c("dodgerblue2", "olivedrab4", "gold")
     } else if (length(unique(melted$time_period)) == 4){
       color_ord = c("grey47", "dodgerblue2", "olivedrab4", "gold")
     } else if (length(unique(melted$time_period)) == 5){
     color_ord = c("red", "grey47", "dodgerblue2", "olivedrab4", "gold")
  }

  melted$cluster <- factor(melted$cluster, levels=region_levels)
  melted$time_period <- factor(melted$time_period, levels=time_label)

  ax_txt_size <- 6; ax_ttl_size <- 8; box_width = 0.4
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
               legend.margin = margin(t=0, r=0, b=0, l=0, unit = 'line'),
               legend.title = element_blank(),
               plot.title = element_text(size=ax_ttl_size, face = "bold",
                                         margin = margin(t=.15, r=.1, b=0, l=0, "line")), # b=-1.5
               plot.subtitle = element_text(size=ax_txt_size, face = "plain"),
               strip.text.x = element_text(size = ax_ttl_size, face = "bold",
                                           margin = margin(.15, 0, .15, 0, "line")),
               axis.ticks.y = element_line(size = .1, color = "black"),
               axis.ticks.x = element_blank(),
               axis.text.y = element_text(size = ax_txt_size, face = "bold", 
                                          color = "black"),
               axis.text.x = element_blank(),
               axis.title.y = element_text(size = ax_ttl_size, face = "bold", 
                                           margin = margin(t=0, r=2, b=0, l=0)),
               axis.title.x = element_blank()
              )

  signif <- if (grepl("diff", tgt_col)) "%1.1f" else "%1.0f"
  ########################################################################
  ######
  ###### plot
  ######
  ggplot(data = melted, aes(x=cluster, y=value, fill=time_period)) +
  the + 
  geom_boxplot(outlier.size = -0.3, notch=F, 
               width = box_width, lwd=.1, 
               position = position_dodge(0.6), outlier.shape=NA
               ) +
  scale_x_discrete(expand=c(0.1, 0)) + 
  xlab("precip. group") +
  ylab(y_lab) +
  scale_fill_manual(values = color_ord, labels = time_label) +
  geom_text(data = medians, 
            aes(label = sprintf(signif, medians$med), y = medians$med), 
            size = 2, fontface = "bold",
            position = position_dodge(.6), vjust = -.4)
}

storm_box_25_sep_clust <- function(dt_25){
  categ_lab <- sort(unique(dt_25$return_period))
  
  if (length(unique(dt_25$return_period)) == 3){
    color_ord = c("dodgerblue2", "olivedrab4", "gold")
    } else if (length(unique(dt_25$return_period)) == 4){
      color_ord = c("grey47", "dodgerblue2", "olivedrab4", "gold")
    } else if (length(unique(dt_25$return_period)) == 5){
    color_ord = c("red", "grey47", "dodgerblue2", "olivedrab4", "gold")
  }

  medians <- data.frame(dt_25) %>% 
             group_by(return_period, emission, cluster) %>% 
             summarise(med_25 = median(twenty_five_years)) %>% 
             data.table()

  melted <- melt(dt_25, 
                 id = c("cluster", "return_period", "emission"))

  ax_txt_size <- 6; ax_ttl_size <- 8; box_width = 0.4
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
               legend.margin = margin(t=0, r=0, b=0, l=0, unit = 'line'),
               legend.title = element_blank(),
               plot.title = element_text(size = ax_ttl_size, face = "bold",
                                         margin = margin(t=.15, r=.1, b=0, l=0, "line")),
               plot.subtitle = element_text(size=ax_txt_size, face = "plain"),
               strip.text.x = element_text(size = ax_ttl_size, face = "bold",
                                           margin = margin(.15, 0, .15, 0, "line")),
               axis.ticks.y = element_line(size = .1, color = "black"),
               axis.ticks.x = element_blank(),
               axis.text.y = element_text(size = ax_txt_size, face = "bold", 
                                          color = "black"),
               axis.text.x = element_blank(),
               axis.title.y = element_text(size = ax_ttl_size, face = "bold", 
                                           margin = margin(t=0, r=2, b=0, l=0)),
               axis.title.x = element_blank()
              )

  box_p <- ggplot(data = melted, 
                  aes(x=cluster, y=value, fill=return_period)) +
           geom_boxplot(outlier.size = -0.3, notch=F, 
                        width = box_width, lwd=.1, 
                        position = position_dodge(0.6), 
                        outlier.shape=NA
                        ) +
           scale_x_discrete(expand=c(0.1, 0)) + 
           the +
           ylab("design storm intensity (mm/hr)") + 
           scale_fill_manual(values = color_ord,
                             name = "Return\nPeriod", 
                             labels = categ_lab) + 
           scale_y_continuous(breaks = seq(0, 20, by=2)) + 
           geom_text(data = medians, 
                    aes(label = sprintf("%1.1f", medians$med), 
                         y = medians$med), 
                    size = 2, fontface = "bold",
                    position = position_dodge(.6), vjust = -.4)
 return(box_p)
}

storm_diff_box_25yr_sep_clust <- function(data_tb, tgt_col){
  data_tb <- data_tb %>% 
             filter(time_interval=="twenty_five_years") %>% 
             data.table()

  needed_cols <- c("return_period", "emission", "cluster", tgt_col)
  data_tb <- subset(data_tb, select=needed_cols)

  time_label <- sort(unique(data_tb$return_period))
  data_tb$return_period <- factor(data_tb$return_period, levels=time_label)
  data_tb$cluster <- factor(data_tb$cluster, 
                            levels=c("Western coastal", 
                                     "Cascade foothills", 
                                     "Northwest Cascades", 
                                     "Northcentral Cascades", 
                                     "Northeast Cascades"))
     
  if (length(time_label) == 3){
    color_ord = c("dodgerblue2", "olivedrab4", "gold")
    } else if (length(time_label) == 4){
      color_ord = c("grey47", "dodgerblue2", "olivedrab4", "gold")
    } else if (length(time_label) == 5){
    color_ord = c("red", "grey47", "dodgerblue2", "olivedrab4", "gold")
  }
  if (tgt_col=="perc_diff"){
     y_labb <- "differences (%)"
     } else {
      y_labb <- "magnitude of differences"
  }
  medians <- data.frame(data_tb) %>% 
             group_by(return_period, emission, cluster) %>% 
             summarise( med = median(get(tgt_col))) %>% 
             data.table()

  ax_txt_size <- 6; ax_ttl_size <- 8; box_width = 0.4
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
               legend.margin = margin(t=0, r=0, b=0, l=0, unit = 'line'),
               legend.title = element_blank(),
               plot.title = element_text(size = ax_ttl_size, face = "bold",
                                         margin = margin(t=.15, r=.1, b=0, l=0, "line")),
               plot.subtitle = element_text(size=ax_txt_size, face = "plain"),
               strip.text.x = element_text(size = ax_ttl_size, face = "bold",
                                           margin = margin(.15, 0, .15, 0, "line")),
               axis.ticks.y = element_line(size = .1, color = "black"),
               axis.ticks.x = element_blank(),
               axis.text.y = element_text(size = ax_txt_size, face = "bold", 
                                          color = "black"),
               axis.text.x = element_blank(),
               axis.title.y = element_text(size = ax_ttl_size, face = "bold", 
                                           margin = margin(t=0, r=2, b=0, l=0)),
               axis.title.x = element_blank()
              )

  ggplot(data = data_tb, 
         aes(x=cluster, y=get(tgt_col), fill=return_period)) +
  geom_hline(yintercept= 0, color = "red", size=.3) + 
  geom_boxplot(outlier.size = -0.3, notch=F, 
               width = box_width, lwd=.1, 
               position = position_dodge(0.6), outlier.shape=NA
               ) +
  scale_x_discrete(expand=c(0.1, 0)) + 
  ylab(y_labb) + 
  scale_fill_manual(values = color_ord,
                    name = "Return\nPeriod", 
                    labels = time_label) + 
  the +
  scale_y_continuous(breaks = seq(-100, 100, by=10)) + 
  geom_text(data = medians, 
            aes(label = sprintf("%1.1f", medians$med), y = medians$med),
            size = 2, fontface = "bold",
            position = position_dodge(.6), vjust = -.4)
}

