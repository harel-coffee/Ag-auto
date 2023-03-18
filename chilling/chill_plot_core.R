
double_cloud <- function(d1){
  ggplot(d1, aes(x=year, y=value, fill=factor(variable))) +
  labs(x = "year", y = "day of year", fill = "data type") +
  guides(fill=guide_legend(title="")) + 
  facet_grid(. ~ emission ~ city, scales="free") +
  # geom_line(aes(fill=factor(Timeframe), color=factor(Timeframe) )) +
  stat_summary(geom="ribbon", fun.y=function(z) { quantile(z,0.5) }, 
                              fun.ymin=function(z) { quantile(z,0) }, 
                              fun.ymax=function(z) { quantile(z,1) }, 
               alpha=0.2) +
  
  stat_summary(geom="ribbon", fun.y=function(z) { quantile(z,0.5) }, 
                              fun.ymin=function(z) { quantile(z,0.1)}, 
                              fun.ymax=function(z) { quantile(z,0.9)}, 
               alpha=0.4) +
  
  stat_summary(geom="ribbon", fun.y=function(z) { quantile(z,0.5) }, 
                              fun.ymin=function(z) { quantile(z,0.25) }, 
                              fun.ymax=function(z) { quantile(z,0.75) }, 
               alpha=0.8) + 
  
  stat_summary(geom="line", fun.y=function(z) { quantile(z,0.5) })+
  scale_color_manual(values=c("darkgreen", "orange"),
                     breaks=c("thresh", "fifty_perc_DoY"),
                     labels=c("CP threshold", "Bloom day"))+
  
  scale_fill_manual(values=c("darkgreen", "orange"),
                     breaks=c("thresh", "fifty_perc_DoY"),
                     labels=c("CP threshold", "Bloom day")) +
  
  scale_x_continuous(breaks=seq(1970, 2100, 10)) +
  
  scale_y_continuous(breaks = chill_doy_map$day_count_since_sept, 
                     labels = chill_doy_map$letter_day) + 
  
  theme(panel.grid.major = element_line(size=0.2),
        panel.spacing=unit(.5, "cm"),
        legend.text=element_text(size=12, face="bold"),
        legend.title = element_blank(),
        legend.position = c(0.1, 0.1),
        strip.text = element_text(face="bold", size=16, color="black"),
        axis.text = element_text(face="bold", size=10, color="black"),
        axis.ticks = element_line(color = "black", size = .2),
        axis.title.x = element_text(face="bold", size=16, 
                                    margin=margin(t=10, r=0, b=0, l=0)),

        axis.title.y = element_text(face="bold", size=16, 
                                    margin=margin(t=0, r=10, b=0, l=0)),

        plot.title = element_text(lineheight=.8, face="bold")
        ) + 
  ylim(0, 260)
}

cloudy_frost <- function(d1, colname="chill_dayofyear", fil){
  if (colname=="fifty_perc_DoY"){
     cls <- "orange"
     } else if(colname == "chill_dayofyear"){
      cls <- "deepskyblue"
      } else {
        cls <- "darkgreen"
  }

  ggplot(d1, aes(x=year, y=get(colname), fill=fil, group=time_period)) +
  labs(x = "year", y = "day of year") + #, fill = "Climate Group"
  # guides(fill=guide_legend(title="Time period")) + 
  facet_grid(. ~ emission ~ city) + # scales = "free"
  stat_summary(geom="ribbon", fun.y=function(z) { quantile(z,0.5) }, 
                              fun.ymin=function(z) { quantile(z,0) }, 
                              fun.ymax=function(z) { quantile(z,1) }, 
               alpha=0.2) +

  stat_summary(geom="ribbon", fun.y=function(z) {quantile(z,0.5) }, 
               fun.ymin=function(z) { quantile(z,0.1) }, 
               fun.ymax=function(z) { quantile(z,0.9) }, alpha=0.4) +

  stat_summary(geom="ribbon", fun.y=function(z) { quantile(z,0.5) }, 
                              fun.ymin=function(z) { quantile(z,0.25) }, 
                              fun.ymax=function(z) { quantile(z,0.75) }, 
               alpha=0.8) +

  stat_summary(geom="line", fun.y=function(z) {quantile(z,0.5) }, 
               size = 1)+       
  scale_x_continuous(breaks=seq(1970, 2100, 10)) +
  scale_y_continuous(breaks = chill_doy_map$day_count_since_sept, 
                     labels = chill_doy_map$letter_day) +
  scale_color_manual(values = cls) +
  scale_fill_manual(values = cls) +
  theme(panel.grid.major = element_line(size=0.2),
        panel.spacing=unit(.5, "cm"),
        legend.text=element_text(size=12, face="bold"),
        legend.title = element_blank(),
        legend.position = c(.1, .1),
        strip.text = element_text(face="bold", size=16, color="black"),
        axis.text = element_text(face="bold", size=10, color="black"),
        axis.ticks = element_line(color = "black", size = .2),
        axis.title.x = element_text(face="bold", size=16, margin=margin(t=10, r=0, b=0, l=0)),
        axis.title.y = element_text(face="bold", size=16, margin=margin(t=0, r=10, b=0, l=0)),
        plot.title = element_text(lineheight=.8, face="bold")
        )
}
####################################################################################
plot_bloom_box <- function(dt, colname, ct){
  color_ord = c("grey40", "dodgerblue", "olivedrab4", "red")
  categ_lab = c("1979-2015", "2026-2050", "2051-2075", "2076-2095")
  if (colname == "cripps_pink"){
      title_ <- paste0("Cripps pink")
     } else if (colname == "gala"){
      title_ <- paste0("Gala")
     } else {
      title_ <- paste0("Red delicious")
  }

  df <- data.frame(dt)
  df <- (df %>% group_by(time_period, emission, city))
  medians <- (df %>% summarise(med = median(get(colname))))

  box_width <- 0.35
  title_ <- paste0(title_, ", ",ct)
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

  safe_b <- ggplot(data = dt, aes(x=time_period, 
                                  y=get(colname), 
                                  fill=time_period)) +
            geom_boxplot(outlier.size = -.05, 
                         notch=F, width=box_width, 
                         lwd=.3, alpha=.8) +
            facet_grid(~ emission) +
            the_theme + 
            scale_fill_manual(values = color_ord, name = "Time\nPeriod", 
                              labels = categ_lab) + 

            scale_color_manual(values = color_ord, labels = categ_lab,
                               name = "Time\nPeriod", limits = color_ord) + 

            scale_x_discrete(breaks = c("1979-2016", "2026-2050", 
                                        "2051-2075", "2076-2095"),
                             labels = categ_lab) +

            ggtitle(lab=title_) +

            geom_text(data = medians, 
                      aes(label = sprintf("%1.0f", medians$med), 
                          y = medians$med), 
                          colour = "black", fontface = "bold",
                          size=5, 
                          position=position_dodge(.09),
                          vjust = 1)
}
##############
############## Time Series
##############
plot_frost_TS <- function(dt, colname, title_){ # This is used for plotting
  if (length(unique(dt$location)) <= 15){
    p <- ggplot(data=dt) + 
         geom_point(aes(x = year, y = get(colname)), alpha = 0.25, shape = 21, size = 1) + 
         geom_line(aes(x = year, y = get(colname), group=model)) + 
         geom_smooth(aes(x = year, y = get(colname), group=model), 
                     method = "lm", size=1, se=F) + 
         facet_grid(~ emission) +
         theme(plot.title = element_text(size=17, face="bold"),
               plot.margin = margin(t=.2, r=.5, b=.2, l=.2, "cm"),
               panel.grid.minor = element_blank(),
               legend.position = "none",
               panel.spacing = unit(.5, "cm"),
               panel.grid.major = element_line(size = 0.1),
               axis.ticks = element_line(color = "black", size = .2),
               strip.text.x = element_text(size = 20, face = "bold"),
               strip.text.y = element_text(size = 20, face = "bold"),
               
               axis.text.x = element_text(size=16, face="plain", 
                                          color="black", angle=-30),
               
               axis.text.y = element_text(size = 16, color="black"),

               axis.title.x = element_blank(),
               axis.title.y = element_blank(), 
               # element_text(size=28, face="bold", 
               #             margin=margin(t=0, r=12, b=0, l=0)),
               # axis.ticks.y = element_blank()
              ) + 
       # labs(y = "day count (starting Sept. 1)") +
         scale_y_continuous(breaks = c(1, 31, 62, 92, 123, 154, 183), 
                            labels = c("Sept. 1", "Oct. 1", "Nov. 1", 
                            "Dec. 1", "Jan. 1", "Feb. 1", "Mar. 1")) + 
         ggtitle(lab=title_)

  	 } else {
      p <- ggplot(data=dt) + 
           geom_point(aes(x = year, y = get(colname)), 
                      alpha = 0.25, shape = 21, size = 1) + 

           geom_line(aes(x = year, y = get(colname), group=model)) + 
           geom_smooth(aes(x = year, y = get(colname), group=model), 
                       method = "lm", size=1, se=F) + 
           
           facet_grid(~ emission) +
           
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
                 axis.title.y = element_blank(), #element_text(size = 28, face = "bold", margin = margin(t=0, r=12, b=0, l=0)),
                 # axis.ticks.y = element_blank()
                 ) + 
           # labs(y = "day count (starting Sept. 1)") +
           scale_y_continuous(breaks = c(1, 31, 62, 92, 123, 154, 183), 
                              labels = c("Sept. 1", "Oct. 1", "Nov. 1", 
                                         "Dec. 1", "Jan. 1", "Feb. 1", 
                                         "Mar. 1")) + 
           ggtitle(lab=title_)
  }
  return (p)
}

boxplot_frost_dayofyear <- function(dt, kth_day, sub_title){
  color_ord = c("grey40", "dodgerblue", "olivedrab4", "red")
  categ_lab = c("1979-2015", "2026-2050", "2051-2075", "2076-2095")

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
                     legend.margin=margin(t=.1,r=0,b=.1,l=0,unit='cm'),
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
                     axis.text.x = element_blank(), 
                     # element_text(size= 12, face = "plain", 
                     #              color="black", angle=-30),
                     axis.text.y = element_text(size=12, color="black"),
                     axis.ticks.x = element_blank())
  
  if (length(unique(dt$location)) <= 15){
    box_width <- 0.35
    title_ <- paste0(title_, ", ",ct)
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
                       strip.text = element_text(size=20, face = "bold"),
                       legend.text=element_text(size=20),
                       axis.title.x = element_blank(),
                       axis.title.y = element_blank(),
                       axis.text.x = element_blank(), # element_text(size= 12, face = "plain", color="black", angle=-30),
                       axis.text.y = element_text(size=12, color="black"),
                       axis.ticks.x = element_blank())
    safe_b <- ggplot(data = dt, aes(x=time_period, y= chill_dayofyear, fill=time_period)) +
              geom_boxplot(outlier.size = -.05, notch=F, width=box_width, lwd=.3, alpha=.8) +
              facet_grid(~ emission) +
              the_theme + 
              scale_fill_manual(values = color_ord, name = "Time\nPeriod", 
                                labels = categ_lab) + 
              scale_color_manual(values = color_ord, labels = categ_lab,
                                 name = "Time\nPeriod", limits = color_ord) + 
              scale_x_discrete(breaks = c("1979-2016", "2026-2050", 
                                          "2051-2075", "2076-2095"),
                               labels = categ_lab) +
              scale_y_continuous(breaks = c(1, 31, 62, 92, 123, 154, 183), 
                                 labels = c("Sept. 1", "Oct. 1", "Nov. 1", 
                                            "Dec. 1", "Jan. 1", "Feb. 1", "Mar. 1")) + 
              ggtitle(lab=title_) +
              geom_text(data = medians, 
                        aes(label = sprintf("%1.0f", medians$med), y=medians$med), 
                            colour = "black", fontface = "bold",
                            size=5, position = position_dodge(.09), vjust = -.4)
      } else {
      	box_width <- 0.25
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
                           axis.title.x = element_blank(),
                           axis.title.y = element_blank(),
                           axis.text.x = element_blank(), # element_text(size= 12, face = "plain", color="black", angle=-30),
                           axis.text.y = element_text(size=12, color="black"),
                           axis.ticks.x = element_blank())
        safe_b <- ggplot(data = dt, aes(x=time_period, y= chill_dayofyear, fill=time_period)) +
                  geom_boxplot(outlier.size = -.05, 
                  	           notch=F, width=box_width, 
                  	           lwd=.3, alpha=.8, position=position_dodge(.5)) +
                  # geom_point(size=.1, shape=21, position=position_jitterdodge()) +
                  facet_grid(~ emission) +
                  the_theme + 
                  scale_fill_manual(values = color_ord, name = "Time\nPeriod", 
                                    labels = categ_lab) + 
                  scale_color_manual(values = color_ord, labels = categ_lab,
                                     name = "Time\nPeriod", limits = color_ord) + 
                  scale_x_discrete(breaks = c("1979-2016", "2026-2050", 
                                              "2051-2075", "2076-2095"),
                                   labels = categ_lab) +
                  scale_y_continuous(breaks = c(1, 31, 62, 92, 123, 154, 183), 
                                     labels = c("Sept. 1", "Oct. 1", "Nov. 1", 
                                                "Dec. 1", "Jan. 1", "Feb. 1", "Mar. 1")) + 
                  ggtitle(lab=title_, subtitle=sub_title) + 
                  geom_text(data = medians, 
                            aes(label = sprintf("%1.0f", medians$med), y=medians$med), 
                                colour = "black", fontface = "bold",
                                size=5, position = position_dodge(.09), vjust = -.4)

  }
  return(safe_b)
}
####################################################################################
safe_box_plot <- function(data, due, chill_start, col = "quan_90"){
  color_ord = c("grey47" , "dodgerblue", "olivedrab4", "red") #

  data$time_period <- as.character(data$time_period)
  if ("2025_2050" %in% unique(data$time_period)){
    data$time_period[data$time_period=="2025_2050"] <- "2026_2050"
  }

  if ("2025-2050" %in% unique(data$time_period)){
    data$time_period[data$time_period=="2025-2050"] <- "2026-2050"
  }


  categ_lab = c("Historical", "2026-2050", "2051-2075", "2076-2099")
  box_width = 0.4
  data$time_period <- gsub("_", "-", data$time_period)
  data$time_period <- factor(data$time_period, levels = categ_lab, order=TRUE)

  df <- data.frame(data)
  df <- df %>% group_by(time_period, emission, city)
  medians <- (df %>% summarise(med = median(get(col))))
  
  the_theme = theme(plot.margin = unit(c(t=.2, r=.2, b=.2, l=0.2), "cm"),
                    panel.border = element_rect(fill=NA, size=.3),
                    panel.grid.major = element_line(size = 0.1),
                    panel.grid.minor = element_blank(),
                    panel.spacing = unit(.25, "cm"),
                    legend.position = "bottom", 
                    legend.key.size = unit(1.5, "line"),
                    legend.spacing.x = unit(.05, 'cm'),
                    panel.spacing.y = unit(.5, 'cm'),
                    legend.text = element_text(size=14),
                    legend.margin = margin(t=0, r=0, b=0, l=0, unit = 'cm'),
                    legend.title = element_blank(),
                    plot.title = element_text(size=14, face = "bold"),
                    plot.subtitle = element_text(face = "bold"),
                    strip.text.x = element_text(size=14, face="bold"),
                    strip.text.y = element_text(size=14, face="bold"),
                    axis.ticks = element_line(size=.1, color="black"),
                    axis.title.y = element_text(size=14, face="bold", margin = margin(t=0, r=10, b=0, l=0)),
                    axis.text.y = element_text(size=10, face="plain", color="black"),
                    axis.text.x = element_blank(),
                    axis.title.x = element_blank()
                    )
  
  safe_apr <- ggplot(data = data, aes(x=time_period, y=get(col), fill=time_period)) +
              geom_boxplot(outlier.size= -.15, notch=F, width=box_width, lwd=.1) +
              labs(x="", y="safe chill") +
              facet_grid(~ emission ~ city ) + 
              the_theme + 
              scale_fill_manual(values = color_ord,
                                name = "Time\nPeriod", 
                                labels = categ_lab) + 
              scale_color_manual(values = color_ord,
                                 name = "Time\nPeriod", 
                                 limits = color_ord,
                                 labels = categ_lab) + 
              scale_x_discrete(breaks = categ_lab,
                               labels = categ_lab)  +
              geom_text(data = medians, 
                        aes(label = sprintf("%1.0f", medians$med), y=medians$med), 
                            size=4.5, 
                            position =  position_dodge(.09),
                            vjust = 0, hjust=.5) # +
            # ggtitle(lab=paste0("Safe chill accumulation")) 
            #, subtitle = paste0("chill season started on ", chill_start)
  
  return(safe_apr)
}

ensemble_map <- function(data, color_col, due) {
  states <- map_data("state")
  states_cluster <- subset(states, region %in% c("oregon", "washington", "idaho"))

  if (color_col=="mean_over_model"){
     low_lim = min(data$mean_over_model)
     up_lim = max(data$mean_over_model)
  } else if (color_col=="median_over_model"){
     low_lim = min(data$median_over_model)
     up_lim = max(data$median_over_model)
  }
  
  data %>% ggplot() +
           geom_polygon(data = states_cluster, aes(x=long, y=lat, group = group),
                        fill = "grey", color = "black") +
            # aes_string to allow naming of column in function 
            geom_point(aes_string(x = "long", y = "lat",
                       color = color_col), alpha = 0.4, size=.4) +
           coord_fixed(xlim = c(-124.5, -111.4),  ylim = c(41, 50.5), ratio = 1.3) +
           facet_grid(~ emission ~ time_period) +
           ggtitle(paste0("Ensemble ", unlist(strsplit(color_col, "_"))[1] , " by ", due, " 1st")) + 
           theme_bw() + 
           theme(legend.position = "bottom",
                 legend.title = element_blank(),
                 legend.key.size = unit(1.4, "line"),
                 plot.margin = margin(t=0, r=0.2, b=0, l=0.2, unit = 'cm')
                 # axis.text.x = element_text(size=3, face="plain", color="black"),
                 # axis.text.y = element_text(size=3, face="plain", color="black"),
                 # legend.margin = margin(t=0, r=0, b=-0.1, l=0, unit = 'cm')
                ) +
           scale_color_gradient2(midpoint=(low_lim + up_lim)/2, low="red", mid="white", high="blue", 
                                 space ="Lab")
}

produce_data_4_plots <- function(data){
  needed_cols = c("chill_season", "sum_J1", "sum_F1", "sum_M1", "sum_A1", 
                  "year", "model", "emission", "lat", "long")

  ################### CLEAN DATA
  data = subset(data, select=needed_cols)
  # pick up future data
  # data = data %>% filter(scenario != "historical")
  data <- organize_non_over_time_period_two_hist(data)

  ################### GENERATE STATS
  #######################################################################
  ##                                                                   ##
  ##   Find the 90th percentile of the chill units                     ##
  ##   Grouped by location, model, time_period and rcp                 ##
  ##   This could be used for box plots, later compute the mean.       ##
  ##   for maps                                                        ##
  ##                                                                   ##
  #######################################################################
  quan_per_loc_period_model_jan <- data %>% 
                                   group_by(time_period, lat, long, scenario, model, warm_cold) %>%
                                   summarise(quan_90 = quantile(sum_J1, probs = 0.1)) %>%
                                   data.table()
  
  quan_per_loc_period_model_feb <- data %>% 
                                   group_by(time_period, lat, long, scenario, model, warm_cold) %>%
                                   summarise(quan_90 = quantile(sum_F1, probs = 0.1)) %>%
                                   data.table()

  quan_per_loc_period_model_mar <- data %>% 
                                   group_by(time_period, lat, long, scenario, model, warm_cold) %>%
                                   summarise(quan_90 = quantile(sum_M1, probs = 0.1)) %>%
                                   data.table()

  quan_per_loc_period_model_apr <- data %>% 
                                   group_by(time_period, lat, long, scenario, model, warm_cold) %>%
                                   summarise(quan_90 = quantile(sum_A1, probs = 0.1)) %>%
                                   data.table()

  # it seems there is a library, perhaps tidyverse, that messes up
  # the above line, so the two variables above are 1-by-1. 
  # just close and re-open R Studio
  ########################################################################
  #######                                                          #######
  #######                     Means                                #######
  #######                                                          #######
  ########################################################################
  
  mean_quan_per_loc_period_model_jan <- quan_per_loc_period_model_jan %>%
                                        group_by(time_period, lat, long, scenario) %>%
                                        summarise(mean_over_model = mean(quan_90)) %>%
                                        data.table()

  mean_quan_per_loc_period_model_feb <- quan_per_loc_period_model_feb %>%
                                        group_by(time_period, lat, long, scenario) %>%
                                        summarise(mean_over_model = mean(quan_90)) %>%
                                        data.table()
  
  mean_quan_per_loc_period_model_mar <- quan_per_loc_period_model_mar %>%
                                        group_by(time_period, lat, long, scenario) %>%
                                        summarise(mean_over_model = mean(quan_90)) %>%
                                        data.table()

  mean_quan_per_loc_period_model_apr <- quan_per_loc_period_model_apr %>%
                                        group_by(time_period, lat, long, scenario) %>%
                                        summarise(mean_over_model = mean(quan_90)) %>%
                                        data.table()
  ########################################################################
  #######                                                          #######
  #######                     Medians                              #######
  #######                                                          #######
  ########################################################################
  median_quan_per_loc_period_model_jan <- quan_per_loc_period_model_jan %>%
                                          group_by(time_period, lat, long, scenario) %>%
                                          summarise(median_over_model = median(quan_90)) %>%
                                          data.table()

  median_quan_per_loc_period_model_feb <- quan_per_loc_period_model_feb %>%
                                          group_by(time_period, lat, long, scenario) %>%
                                          summarise(median_over_model = median(quan_90)) %>%
                                          data.table()
  
  median_quan_per_loc_period_model_mar <- quan_per_loc_period_model_mar %>%
                                          group_by(time_period, lat, long, scenario) %>%
                                          summarise(median_over_model = median(quan_90)) %>%
                                          data.table()
  
  median_quan_per_loc_period_model_apr <- quan_per_loc_period_model_apr %>%
                                          group_by(time_period, lat, long, scenario) %>%
                                          summarise(median_over_model = median(quan_90)) %>%
                                          data.table()
  
  # quan_per_loc_period_model_jan$time_period = factor(quan_per_loc_period_model_jan$time_period, order=T)
  # mean_quan_per_loc_period_model_jan$time_period = factor(mean_quan_per_loc_period_model_jan$time_period, order=T)
  # median_quan_per_loc_period_model_jan$time_period = factor(median_quan_per_loc_period_model_jan$time_period, order=T)

  # quan_per_loc_period_model_feb$time_period_feb = factor(quan_per_loc_period_model_feb$time_period, order=T)
  # mean_quan_per_loc_period_model_feb$time_period = factor(mean_quan_per_loc_period_model_feb$time_period, order=T)
  # median_quan_per_loc_period_model_feb$time_period= factor(median_quan_per_loc_period_model_feb$time_period, order=T)

  return(list(quan_per_loc_period_model_jan,
              mean_quan_per_loc_period_model_jan,
              median_quan_per_loc_period_model_jan,
              quan_per_loc_period_model_feb,
              mean_quan_per_loc_period_model_feb,
              median_quan_per_loc_period_model_feb,
              quan_per_loc_period_model_mar,
              mean_quan_per_loc_period_model_mar,
              median_quan_per_loc_period_model_mar,
              quan_per_loc_period_model_apr,
              mean_quan_per_loc_period_model_apr,
              median_quan_per_loc_period_model_apr
              )
        )
}

produce_data_4_safe_chill_box_plots_new_seasons <- function(data){

  ################### CLEAN DATA
  data <- organize_non_over_time_period_two_hist(data)

  ################### GENERATE STATS
  #######################################################################
  ##                                                                   ##
  ##   Find the 90th percentile of the chill units                     ##
  ##   Grouped by location, model, time_period and rcp                 ##
  ##   This could be used for box plots, later compute the mean.       ##
  ##   for maps                                                        ##
  ##                                                                   ##
  #######################################################################
  quan_per_loc_period_model_jan <- data %>% 
                                   group_by(time_period, lat, long, scenario, model, start) %>%
                                   summarise(quan_90 = quantile(sum_J1, probs = 0.1)) %>%
                                   data.table()
  
  quan_per_loc_period_model_feb <- data %>% 
                                   group_by(time_period, lat, long, scenario, model, start) %>%
                                   summarise(quan_90 = quantile(sum_F1, probs = 0.1)) %>%
                                   data.table()

  quan_per_loc_period_model_mar <- data %>% 
                                   group_by(time_period, lat, long, scenario, model, start) %>%
                                   summarise(quan_90 = quantile(sum_M1, probs = 0.1)) %>%
                                   data.table()

  quan_per_loc_period_model_apr <- data %>% 
                                   group_by(time_period, lat, long, scenario, model, start) %>%
                                   summarise(quan_90 = quantile(sum_A1, probs = 0.1)) %>%
                                   data.table()

  # it seems there is a library, perhaps tidyverse, that messes up
  # the above line, so the two variables above are 1-by-1. 
  # just close and re-open R Studio
  ########################################################################
  #######                                                          #######
  #######                     Means                              #######
  #######                                                          #######
  ########################################################################
  
  mean_quan_per_loc_period_model_jan <- quan_per_loc_period_model_jan %>%
                                        group_by(time_period, lat, long, scenario, start) %>%
                                        summarise(mean_over_model = mean(quan_90)) %>%
                                        data.table()

  mean_quan_per_loc_period_model_feb <- quan_per_loc_period_model_feb %>%
                                        group_by(time_period, lat, long, scenario, start) %>%
                                        summarise(mean_over_model = mean(quan_90)) %>%
                                        data.table()
  
  mean_quan_per_loc_period_model_mar <- quan_per_loc_period_model_mar %>%
                                        group_by(time_period, lat, long, scenario, start) %>%
                                        summarise(mean_over_model = mean(quan_90)) %>%
                                        data.table()

  mean_quan_per_loc_period_model_apr <- quan_per_loc_period_model_apr %>%
                                        group_by(time_period, lat, long, scenario, start) %>%
                                        summarise(mean_over_model = mean(quan_90)) %>%
                                        data.table()
  ########################################################################
  #######                                                          #######
  #######                     Medians                              #######
  #######                                                          #######
  ########################################################################
  median_quan_per_loc_period_model_jan <- quan_per_loc_period_model_jan %>%
                                          group_by(time_period, lat, long, scenario, start) %>%
                                          summarise(median_over_model = median(quan_90)) %>%
                                          data.table()

  median_quan_per_loc_period_model_feb <- quan_per_loc_period_model_feb %>%
                                          group_by(time_period, lat, long, scenario, start) %>%
                                          summarise(median_over_model = median(quan_90)) %>%
                                          data.table()
  
  median_quan_per_loc_period_model_mar <- quan_per_loc_period_model_mar %>%
                                          group_by(time_period, lat, long, scenario, start) %>%
                                          summarise(median_over_model = median(quan_90)) %>%
                                          data.table()
  
  median_quan_per_loc_period_model_apr <- quan_per_loc_period_model_apr %>%
                                          group_by(time_period, lat, long, scenario, start) %>%
                                          summarise(median_over_model = median(quan_90)) %>%
                                          data.table()

  return(list(quan_per_loc_period_model_jan,
              mean_quan_per_loc_period_model_jan,
              median_quan_per_loc_period_model_jan,
              quan_per_loc_period_model_feb,
              mean_quan_per_loc_period_model_feb,
              median_quan_per_loc_period_model_feb,
              quan_per_loc_period_model_mar,
              mean_quan_per_loc_period_model_mar,
              median_quan_per_loc_period_model_mar,
              quan_per_loc_period_model_apr,
              mean_quan_per_loc_period_model_apr,
              median_quan_per_loc_period_model_apr
              )
        )
}

organize_non_over_time_period_two_hist <- function(data){
  data = data %>% filter(year <=2005 | year>=2025)
  time_periods = c("Historical","2025_2050", "2051_2075", "2076_2099")
  
  data$time_period = 0L
  data$time_period[data$year <= 2005] = time_periods[1]
  data$time_period[data$year >= 2025 & data$year <= 2050] = time_periods[2]
  data$time_period[data$year >= 2051 & data$year <= 2075] = time_periods[3]
  data$time_period[data$year >= 2076] = time_periods[4]
  
  data$time_period = factor(data$time_period, levels = time_periods, order=T)
  data_f <- data %>% filter(time_period != "Historical")
  
  data_h_rcp85 <- data %>% filter(time_period == "Historical")
  data_h_rcp45 <- data %>% filter(time_period == "Historical")
  
  data_h_rcp85$scenario = "RCP 8.5"
  data_h_rcp45$scenario = "RCP 4.5"
  
  data_f$scenario[data_f$scenario=="rcp85"] = "RCP 8.5"
  data_f$scenario[data_f$scenario=="rcp45"] = "RCP 4.5"

  data = rbind(data_f, data_h_rcp45, data_h_rcp85)
  rm(data_h_rcp45, data_h_rcp85, data_f)
  return(data)
}

####################################################################################
############
############
############             Written on March 5th to fill in the paper:
############
############
####################################################################################

organize_time_period_two_hist_w_only_observed <- function(data){
  data$time_period[data$time_period == "1979-2015"] <- "Historical"
  time_periods = c("Historical","2026-2050", "2051-2075", "2076-2099")
  data$time_period = factor(data$time_period, levels = time_periods, order=T)
  
  data_f <- data %>% filter(time_period != "Historical")
  
  data_h_rcp85 <- data %>% filter(time_period == "Historical")
  data_h_rcp45 <- data %>% filter(time_period == "Historical")
  
  data_h_rcp85$emission = "RCP 8.5"
  data_h_rcp45$emission = "RCP 4.5"
  
  data_f$emission[data_f$emission=="rcp85"] = "RCP 8.5"
  data_f$emission[data_f$emission=="rcp45"] = "RCP 4.5"

  data = rbind(data_f, data_h_rcp45, data_h_rcp85)
  rm(data_h_rcp45, data_h_rcp85, data_f)
  return(data)
}

produce_data_4_plots_w_noly_observed <- function(data){
  needed_cols = c("chill_season", "sum_A1", "year", "model", 
                  "emission", "lat", "long", "location", "time_period")

  ################### CLEAN DATA
  data = subset(data, select=needed_cols)
  # pick up future data
  # data = data %>% filter(emission != "historical")
  data <- organize_time_period_two_hist_w_only_observed(data)

  ################### GENERATE STATS
  #######################################################################
  ##                                                                   ##
  ##   Find the 90th percentile of the chill units                     ##
  ##   Grouped by location, model, time_period and rcp                 ##
  ##   This could be used for box plots, later compute the mean.       ##
  ##   for maps                                                        ##
  ##                                                                   ##
  #######################################################################

  quan_per_loc_period_model_apr <- data %>% 
                                   group_by(time_period, emission, model, location) %>%
                                   summarise(quan_90 = quantile(sum_A1, probs = 0.1)) %>%
                                   data.table()

  # it seems there is a library, perhaps tidyverse, that messes up
  # the above line, so the two variables above are 1-by-1. 
  # just close and re-open R Studio
  ########################################################################
  #######                                                          #######
  #######                     Means                              #######
  #######                                                          #######
  ########################################################################
  
  mean_quan_per_loc_period_model_apr <- quan_per_loc_period_model_apr %>%
                                        group_by(time_period, location, emission) %>%
                                        summarise(mean_over_model = mean(quan_90)) %>%
                                        data.table()
  ########################################################################
  #######                                                          #######
  #######                     Medians                              #######
  #######                                                          #######
  ########################################################################
  
  median_quan_per_loc_period_model_apr <- quan_per_loc_period_model_apr %>%
                                          group_by(time_period, location, emission) %>%
                                          summarise(median_over_model = median(quan_90)) %>%
                                          data.table()
  
  # quan_per_loc_period_model_jan$time_period = factor(quan_per_loc_period_model_jan$time_period, order=T)
  # mean_quan_per_loc_period_model_jan$time_period = factor(mean_quan_per_loc_period_model_jan$time_period, order=T)
  # median_quan_per_loc_period_model_jan$time_period = factor(median_quan_per_loc_period_model_jan$time_period, order=T)

  # quan_per_loc_period_model_feb$time_period_feb = factor(quan_per_loc_period_model_feb$time_period, order=T)
  # mean_quan_per_loc_period_model_feb$time_period = factor(mean_quan_per_loc_period_model_feb$time_period, order=T)
  # median_quan_per_loc_period_model_feb$time_period= factor(median_quan_per_loc_period_model_feb$time_period, order=T)

  return(list(quan_per_loc_period_model_apr,
              mean_quan_per_loc_period_model_apr,
              median_quan_per_loc_period_model_apr
              )
        )
}


safe_box_plot_per_city <- function(data, due, chill_start){
  color_ord = c("grey47" , "dodgerblue", "olivedrab4", "red") #

  data$time_period <- as.character(data$time_period)
  if ("2025_2050" %in% unique(data$time_period)){
    data$time_period[data$time_period=="2025_2050"] <- "2026_2050"
  }

  if ("2025-2050" %in% unique(data$time_period)){
    data$time_period[data$time_period=="2025-2050"] <- "2026-2050"
  }


  categ_lab = c("Historical", "2026-2050", "2051-2075", "2076-2099")
  box_width = 0.4
  data$time_period <- gsub("_", "-", data$time_period)
  data$time_period <- factor(data$time_period, levels = categ_lab, order=TRUE)

  df <- data.frame(data)
  df <- df %>% group_by(time_period, emission, city)
  medians <- (df %>% summarise(med = median(quan_90)))
  
  the_theme = theme(plot.margin = unit(c(t=.2, r=.2, b=.2, l=0.2), "cm"),
                    panel.border = element_rect(fill=NA, size=.3),
                    panel.grid.major = element_line(size = 0.1),
                    panel.grid.minor = element_blank(),
                    panel.spacing = unit(.25, "cm"),
                    legend.position = "bottom", 
                    legend.key.size = unit(1.2, "line"),
                    legend.spacing.x = unit(.05, 'cm'),
                    panel.spacing.y = unit(.5, 'cm'),
                    legend.text = element_text(size=12),
                    legend.margin = margin(t=0, r=0, b=0, l=0, unit = 'cm'),
                    legend.title = element_blank(),
                    plot.title = element_text(size=14, face = "bold"),
                    plot.subtitle = element_text(face = "bold"),
                    strip.text.x = element_text(size=14, face="bold"),
                    strip.text.y = element_text(size=14, face="bold"),
                    axis.ticks = element_line(size=.1, color="black"),
                    axis.title.y = element_text(size=14, face="bold", margin = margin(t=0, r=10, b=0, l=0)),
                    axis.text.y = element_text(size=10, face="plain", color="black"),
                    axis.text.x = element_blank(),
                    axis.title.x = element_blank()
                    )
  
  safe_apr <- ggplot(data = data, aes(x=time_period, y=quan_90, fill=time_period)) +
              geom_boxplot(outlier.size=-.25, notch=F, width=box_width, lwd=.1) +
              labs(x="", y="safe chill") +
              facet_grid(~ emission ~ city ) + 
              the_theme + 
              scale_fill_manual(values = color_ord,
                                name = "Time\nPeriod", 
                                labels = categ_lab) + 
              scale_color_manual(values = color_ord,
                                 name = "Time\nPeriod", 
                                 limits = color_ord,
                                 labels = categ_lab) + 
              scale_x_discrete(breaks = categ_lab,
                               labels = categ_lab)  +
              geom_text(data = medians, 
                        aes(label = sprintf("%1.0f", medians$med), y=medians$med), 
                            size=4.5, 
                            position =  position_dodge(.09),
                            vjust = 0, hjust=.5) # +
            # ggtitle(lab=paste0("Safe chill accumulation")) 
            #, subtitle = paste0("chill season started on ", chill_start)
  
  return(safe_apr)
}

diff_SC_map_one_emission <- function(data, color_col) {
  states <- map_data("state")
  states_cluster <- subset(states, region %in% c("oregon", "washington", "idaho"))

  low_lim = min(data$SC_diff_median)
  up_lim = max(data$SC_diff_median)

  x <- data$location
  x <- sapply(x, 
             function(x) strsplit(x, "_")[[1]], 
              USE.NAMES=FALSE)
  lat = x[1, ]
  long = x[2, ]
  
  data$lat <- as.numeric(lat)
  data$long <- as.numeric(long)
  data <- data.table(data)

  data %>% ggplot() +
           geom_polygon(data = states_cluster, aes(x=long, y=lat, group = group),
                        fill = "grey", color = "black") +
            # aes_string to allow naming of column in function 
           geom_point(aes_string(x = "long", y = "lat", color = color_col), alpha = 0.4, size=.1) +
           coord_fixed(xlim = c(-124.5, -111.4),  ylim = c(41, 50.5), ratio = 1.3) +
           facet_grid(~ emission ~ time_period) +
           # ggtitle(paste0("CP accumulation percentage differeces between projections and observed data")) +
           theme(axis.title.y = element_blank(),
                 axis.title.x = element_blank(),
                 axis.ticks.y = element_blank(), 
                 axis.ticks.x = element_blank(),
                 axis.text.x = element_blank(),
                 axis.text.y = element_blank(),
                 panel.grid.major = element_line(size = 0.1),
                 legend.position="bottom", 
                 legend.title = element_blank(),
                 strip.text = element_text(size=12, face="bold"),
                 plot.margin = margin(t=-0.5, r=0.2, b=-0.5, l=0.2, unit = 'cm')
                 ) + 

           # theme_bw() + 
           # theme(legend.position = "bottom",
           #       legend.title = element_blank(),
           #       legend.key.size = unit(1, "line"),
           #       legend.text=element_text(size=6),
           #       plot.margin = margin(t=0.1, r=0.2, b=0.1, l=0.2, unit = 'cm'),
           #       panel.grid.major = element_line(size = 0.1, linetype = 'solid', colour = "grey"),
           #       panel.grid.minor = element_line(size = 0.1, linetype = 'solid', colour = "grey"),
           #       axis.title.y = element_blank(),
           #       axis.title.x = element_blank(),
           #       axis.ticks.y = element_blank(), 
           #       axis.ticks.x = element_blank(),
           #       axis.text = element_blank(),
           #       plot.title=element_text(size=10, face="bold"),
           #       strip.text = element_text(size=8, face="bold"),
           #       # axis.text.x = element_text(size=3, face="plain", color="black"),
           #       # axis.text.y = element_text(size=3, face="plain", color="black"),
           #       legend.margin = margin(t=0, r=0, b=-0.1, l=0, unit = 'cm')
           #      ) +
           # scale_color_gradient2(midpoint=(low_lim + up_lim)/2, low="red", mid="white", high="blue", 
           #                       space ="Lab") 
           scale_color_gradient2(midpoint = 0, mid = "white", 
                                 high = "blue", low = "red", 
                                 guide = "colourbar", space = "Lab",
                                 limit = c(low_lim, up_lim+10),
                                 breaks = c(-150, -100, -50, 0, 50)) 
}

diff_CP_map_one_emission <- function(data, color_col) {
  states <- map_data("state")
  states_cluster <- subset(states, region %in% c("oregon", "washington", "idaho"))

  if (color_col=="CP_diff_median"){
     low_lim = min(data$CP_diff_median)
     up_lim = max(data$CP_diff_median)
  } else if (color_col=="median_over_model"){
     low_lim = min(data$median_over_model)
     up_lim = max(data$median_over_model)
  }

  x <- data$location
  x <- sapply(x, 
             function(x) strsplit(x, "_")[[1]], 
              USE.NAMES=FALSE)
  lat = x[1, ]
  long = x[2, ]
  
  data$lat <- as.numeric(lat)
  data$long <- as.numeric(long)
  data <- data.table(data)

  data %>% ggplot() +
           geom_polygon(data = states_cluster, aes(x=long, y=lat, group = group),
                        fill = "grey", color = "black") +
            # aes_string to allow naming of column in function 
           geom_point(aes_string(x = "long", y = "lat", color = color_col), alpha = 0.4, size=.1) +
           coord_fixed(xlim = c(-124.5, -111.4),  ylim = c(41, 50.5), ratio = 1.3) +
           facet_grid(~ emission ~ time_period) +
           # ggtitle(paste0("CP accumulation percentage differeces between projections and observed data")) +
           theme(axis.title.y = element_blank(),
                 axis.title.x = element_blank(),
                 axis.ticks.y = element_blank(), 
                 axis.ticks.x = element_blank(),
                 axis.text.x = element_blank(),
                 axis.text.y = element_blank(),
                 panel.grid.major = element_line(size = 0.1),
                 legend.position="bottom", 
                 legend.title = element_blank(),
                 strip.text = element_text(size=12, face="bold"),
                 plot.margin = margin(t=-0.5, r=0.2, b=-0.5, l=0.2, unit = 'cm')
                 ) + 

           # theme_bw() + 
           # theme(legend.position = "bottom",
           #       legend.title = element_blank(),
           #       legend.key.size = unit(.5, "line"),
           #       panel.grid.major = element_line(size = 0.1, linetype = 'solid', colour = "grey"),
           #       panel.grid.minor = element_line(size = 0.1, linetype = 'solid', colour = "grey"),
           #       legend.text=element_text(size=4),
           #       plot.margin = margin(t=0.05, r=0.1, b=0.05, l=0.1, unit = 'cm'),
           #       axis.title.y = element_blank(),
           #       axis.title.x = element_blank(),
           #       axis.ticks.y = element_blank(), 
           #       axis.ticks.x = element_blank(),
           #       axis.text = element_blank(),
           #       plot.title=element_text(size=10, face="bold"),
           #       strip.text = element_text(size=8, face="bold"),
           #       # axis.text.x = element_text(size=3, face="plain", color="black"),
           #       # axis.text.y = element_text(size=3, face="plain", color="black"),
           #       legend.margin = margin(t=-.1, r=0, b=-0.1, l=0, unit = 'cm')
           #      ) +
           # scale_color_gradient2(midpoint=(low_lim + up_lim)/2, low="red", mid="white", high="blue", 
           #                       space ="Lab") 
           scale_color_gradient2(midpoint = 0, mid = "white", 
                                 high = "blue", low = "red", 
                                 guide = "colourbar", space = "Lab",
                                 limit = c(low_lim, up_lim),
                                 breaks = c(-200, -75, -40, -25, 0, 25, 40, 75, 200))
}



diff_CP_map <- function(data, color_col) {
  states <- map_data("state")
  states_cluster <- subset(states, region %in% c("oregon", "washington", "idaho"))

  if (color_col=="CP_diff_median"){
     low_lim = min(data$CP_diff_median)
     up_lim = max(data$CP_diff_median)
  } else if (color_col=="median_over_model"){
     low_lim = min(data$median_over_model)
     up_lim = max(data$median_over_model)
  }

  x <- data$location
  x <- sapply(x, 
             function(x) strsplit(x, "_")[[1]], 
              USE.NAMES=FALSE)
  lat = x[1, ]
  long = x[2, ]
  
  data$lat <- as.numeric(lat)
  data$long <- as.numeric(long)
  data <- data.table(data)

  data %>% ggplot() +
           geom_polygon(data = states_cluster, aes(x=long, y=lat, group = group),
                        fill = "grey", color = "black") +
            # aes_string to allow naming of column in function 
           geom_point(aes_string(x = "long", y = "lat", color = color_col), alpha = 0.4, size=.1) +
           coord_fixed(xlim = c(-124.5, -111.4),  ylim = c(41, 50.5), ratio = 1.3) +
           facet_grid(~ emission ~ time_period) +
           # ggtitle(paste0("CP accumulation percentage differeces between projections and observed data")) +
           theme(axis.title.y = element_blank(),
                 axis.title.x = element_blank(),
                 axis.ticks.y = element_blank(), 
                 axis.ticks.x = element_blank(),
                 axis.text.x = element_blank(),
                 axis.text.y = element_blank(),
                 panel.grid.major = element_line(size = 0.1),
                 legend.position="bottom", 
                 legend.title = element_blank(),
                 strip.text = element_text(size=12, face="bold"),
                 plot.margin = margin(t=-0.5, r=0.2, b=-0.5, l=0.2, unit = 'cm')
                 ) + 

           # theme_bw() + 
           # theme(legend.position = "bottom",
           #       legend.title = element_blank(),
           #       legend.key.size = unit(1, "line"),
           #       legend.text=element_text(size=6),
           #       plot.margin = margin(t=0.1, r=0.2, b=0.1, l=0.2, unit = 'cm'),
           #       panel.grid.major = element_line(size = 0.1, linetype = 'solid', colour = "grey"),
           #       panel.grid.minor = element_line(size = 0.1, linetype = 'solid', colour = "grey"),
           #       axis.title.y = element_blank(),
           #       axis.title.x = element_blank(),
           #       axis.ticks.y = element_blank(), 
           #       axis.ticks.x = element_blank(),
           #       axis.text = element_blank(),
           #       plot.title=element_text(size=10, face="bold"),
           #       strip.text = element_text(size=10, face="bold"),
           #       # axis.text.x = element_text(size=3, face="plain", color="black"),
           #       # axis.text.y = element_text(size=3, face="plain", color="black"),
           #       legend.margin = margin(t=0, r=0, b=-0.1, l=0, unit = 'cm')
           #      ) +
           # scale_color_gradient2(midpoint=(low_lim + up_lim)/2, low="red", mid="white", high="blue", 
           #                       space ="Lab") 
           scale_color_gradient2(midpoint = 0, mid = "white", 
                                 high = "blue", low = "red", 
                                 guide = "colourbar", space = "Lab",
                                 limit = c(low_lim, up_lim),
                                 breaks = c(-200, -75, -40, -25, 0, 25, 40, 75, 200))
}

diff_SC_map <- function(data, color_col) {
  states <- map_data("state")
  states_cluster <- subset(states, region %in% c("oregon", "washington", "idaho"))

  low_lim = min(data$SC_diff_median)
  up_lim = max(data$SC_diff_median)

  x <- data$location
  x <- sapply(x, 
             function(x) strsplit(x, "_")[[1]], 
              USE.NAMES=FALSE)
  lat = x[1, ]
  long = x[2, ]
  
  data$lat <- as.numeric(lat)
  data$long <- as.numeric(long)
  data <- data.table(data)

  data %>% ggplot() +
           geom_polygon(data = states_cluster, aes(x=long, y=lat, group = group),
                        fill = "grey", color = "black") +
            # aes_string to allow naming of column in function 
           geom_point(aes_string(x = "long", y = "lat", color = color_col), alpha = 0.4, size=.1) +
           coord_fixed(xlim = c(-124.5, -111.4),  ylim = c(41, 50.5), ratio = 1.3) +
           facet_grid(~ emission ~ time_period) +
           # ggtitle(paste0("CP accumulation percentage differeces between projections and observed data")) +
           theme(axis.title.y = element_blank(),
                 axis.title.x = element_blank(),
                 axis.ticks.y = element_blank(), 
                 axis.ticks.x = element_blank(),
                 axis.text.x = element_blank(),
                 axis.text.y = element_blank(),
                 panel.grid.major = element_line(size = 0.1),
                 legend.position="bottom", 
                 legend.title = element_blank(),
                 strip.text = element_text(size=12, face="bold"),
                 plot.margin = margin(t=-0.5, r=0.2, b=0, l=0.2, unit = 'cm')
                 ) + 
           
           # theme_bw() + 
           # theme(legend.position = "bottom",
           #       legend.title = element_blank(),
           #       legend.key.size = unit(1, "line"),
           #       legend.text=element_text(size=6),
           #       plot.margin = margin(t=0.1, r=0.2, b=0.1, l=0.2, unit = 'cm'),
           #       panel.grid.major = element_line(size = 0.1, linetype = 'solid', colour = "grey"),
           #       panel.grid.minor = element_line(size = 0.1, linetype = 'solid', colour = "grey"),
           #       axis.title.y = element_blank(),
           #       axis.title.x = element_blank(),
           #       axis.ticks.y = element_blank(), 
           #       axis.ticks.x = element_blank(),
           #       axis.text = element_blank(),
           #       plot.title=element_text(size=10, face="bold"),
           #       strip.text = element_text(size=10, face="bold"),
           #       # axis.text.x = element_text(size=3, face="plain", color="black"),
           #       # axis.text.y = element_text(size=3, face="plain", color="black"),
           #       legend.margin = margin(t=0, r=0, b=-0.1, l=0, unit = 'cm')
           #      ) +
           # scale_color_gradient2(midpoint=(low_lim + up_lim)/2, low="red", mid="white", high="blue", 
           #                       space ="Lab") 
           scale_color_gradient2(midpoint = 0, mid = "white", 
                                 high = "blue", low = "red", 
                                 guide = "colourbar", space = "Lab",
                                 limit = c(low_lim, up_lim+10),
                                 breaks = c(-150, -100, -50, 0, 50)) 
}


acc_CP <- function(data, colname){
  color_ord = c("grey47" , "dodgerblue", "olivedrab4", "red") #

  if (colname == "sum_J1"){
      dd <- paste0("Dec. 31")
     } else if (colname == "sum_F1"){
      dd <- paste0("Jan. 31")
     } else if (colname == "sum_M1"){
      dd <- paste0("Feb. 28/29")
     } else if (colname == "sum_A1"){
      dd <- "Mar. 31"
  }

  data$time_period <- as.character(data$time_period)
  if ("2025_2050" %in% unique(data$time_period)){
    data$time_period[data$time_period=="2025_2050"] <- "2026_2050"
  }

  if ("2025-2050" %in% unique(data$time_period)){
    data$time_period[data$time_period=="2025-2050"] <- "2026-2050"
  }

  categ_lab = c("Historical", "2026-2050", "2051-2075", "2076-2099")
  box_width = 0.6
  data$time_period <- gsub("_", "-", data$time_period)
  data$time_period <- factor(data$time_period, levels = categ_lab, order=TRUE)

  df <- data.frame(data)
  df <- df %>% group_by(time_period, emission, city)
  medians <- (df %>% summarise(med = median(get(colname))))
  
  the_theme = theme(plot.margin = unit(c(t=.2, r=.2, b=.2, l=0.2), "cm"),
                    panel.border = element_rect(fill=NA, size=.3),
                    panel.grid.major = element_line(size = 0.1),
                    panel.grid.minor = element_blank(),
                    panel.spacing = unit(.25, "cm"),
                    legend.position = "bottom", 
                    legend.key.size = unit(2, "line"),
                    legend.spacing.x = unit(.05, 'cm'),
                    panel.spacing.y = unit(.5, 'cm'),
                    legend.text = element_text(size=16),
                    legend.margin = margin(t=0, r=0, b=0, l=0, unit = 'cm'),
                    legend.title = element_blank(),
                    plot.title = element_text(size=14, face = "bold"),
                    plot.subtitle = element_text(face = "bold"),
                    strip.text.x = element_text(size=14, face="bold"),
                    strip.text.y = element_text(size=14, face="bold"),
                    axis.ticks = element_line(size=.1, color="black"),
                    axis.title.y = element_text(size=14, face="bold", margin = margin(t=0, r=10, b=0, l=0)),
                    axis.text.y = element_text(size=14, face="plain", color="black"),
                    axis.text.x = element_blank(),
                    axis.title.x = element_blank()
                    )
  
  CP_b <- ggplot(data = data, aes(x=time_period, y=get(colname), fill=time_period)) +
              geom_boxplot(outlier.size=-.25, notch=F, width=box_width, lwd=.1) +
              labs(x="", y="accumulated chill portions") +
              facet_grid(~ emission ~ city ) + 
              the_theme + 
              scale_fill_manual(values = color_ord,
                                name = "Time\nPeriod", 
                                labels = categ_lab) + 
              scale_color_manual(values = color_ord,
                                 name = "Time\nPeriod", 
                                 limits = color_ord,
                                 labels = categ_lab) + 
              scale_x_discrete(breaks = categ_lab,
                               labels = categ_lab)  +
              geom_text(data = medians, 
                        aes(label = sprintf("%1.0f", medians$med), y=medians$med), 
                            size=6, 
                            position =  position_dodge(.09),
                            vjust = 0, hjust=.5) # +
            # ggtitle(lab=paste0("Safe chill accumulation")) 
            #, subtitle = paste0("chill season started on ", chill_start)
  
  return(CP_b)
}





acc_CP_2rows <- function(data, colname){
  color_ord = c("grey47" , "dodgerblue", "olivedrab4", "red") #

  if (colname == "sum_J1"){
      dd <- paste0("Dec. 31")
     } else if (colname == "sum_F1"){
      dd <- paste0("Jan. 31")
     } else if (colname == "sum_M1"){
      dd <- paste0("Feb. 28/29")
     } else if (colname == "sum_A1"){
      dd <- "Mar. 31"
  }

  data$time_period <- as.character(data$time_period)
  if ("2025_2050" %in% unique(data$time_period)){
    data$time_period[data$time_period=="2025_2050"] <- "2026_2050"
  }

  if ("2025-2050" %in% unique(data$time_period)){
    data$time_period[data$time_period=="2025-2050"] <- "2026-2050"
  }

  categ_lab = c("Historical", "2026-2050", "2051-2075", "2076-2099")
  box_width = 0.4
  data$time_period <- gsub("_", "-", data$time_period)
  data$time_period <- factor(data$time_period, levels = categ_lab, order=TRUE)

  df <- data.frame(data)
  df <- df %>% group_by(time_period, city)
  medians <- (df %>% summarise(med = median(get(colname))))
  
  the_theme = theme(plot.margin = unit(c(t=.2, r=.2, b=.2, l=0.2), "cm"),
                    panel.border = element_rect(fill=NA, size=.3),
                    panel.grid.major = element_line(size = 0.1),
                    panel.grid.minor = element_blank(),
                    panel.spacing = unit(.25, "cm"),
                    legend.position = "bottom", 
                    legend.key.size = unit(1.2, "line"),
                    legend.spacing.x = unit(.05, 'cm'),
                    panel.spacing.y = unit(.5, 'cm'),
                    legend.text = element_text(size=12),
                    legend.margin = margin(t=0, r=0, b=0, l=0, unit = 'cm'),
                    legend.title = element_blank(),
                    plot.title = element_text(size=12, face = "bold"),
                    plot.subtitle = element_text(face = "bold"),
                    strip.text.x = element_text(size=12, face="bold"),
                    strip.text.y = element_text(size=12, face="bold"),
                    axis.ticks = element_line(size=.1, color="black"),
                    axis.title.y = element_text(size=12, face="bold", margin = margin(t=0, r=10, b=0, l=0)),
                    axis.text.y = element_text(size=12, face="plain", color="black"),
                    axis.text.x = element_blank(),
                    axis.title.x = element_blank()
                    )
  
  CP_b <- ggplot(data = data, aes(x=time_period, y=get(colname), fill=time_period)) +
              geom_boxplot(outlier.size=-.25, notch=F, width=box_width, lwd=.1) +
              labs(x="", y="accumulated chill portions") +
              facet_wrap( . ~ city ) + 
              the_theme + 
              scale_fill_manual(values = color_ord,
                                name = "Time\nPeriod", 
                                labels = categ_lab) + 
              scale_color_manual(values = color_ord,
                                 name = "Time\nPeriod", 
                                 limits = color_ord,
                                 labels = categ_lab) + 
              scale_x_discrete(breaks = categ_lab,
                               labels = categ_lab)  +
              geom_text(data = medians, 
                        aes(label = sprintf("%1.0f", medians$med), y=medians$med), 
                            size=4, 
                            position =  position_dodge(.09),
                            vjust = 0, hjust=.5) # +
            # ggtitle(lab=paste0("Safe chill accumulation")) 
            #, subtitle = paste0("chill season started on ", chill_start)
  
  return(CP_b)
}
