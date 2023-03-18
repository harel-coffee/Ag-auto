# panel.grid.major = element_line(size=0.2), # inside theme
# panel.grid.minor = element_blank(),

double_cloud_2_rows_Accum_VertDD_CP_NoSlopes_200TrySpringerSub2_XCut <- function(d1, full_CP_intcpt=73.3, 
                                                                                 heat_intcpt=25, xMax=215){
  #
  # set the followings in a way that x-axis and y-axis have
  # the same length so that the slopes makes sense visually
  #
  ymin <- 0
  ymax <- 800

  xmin <- 1
  xmax <- xMax
   
  newdf <- data.table(chill_doy_map)
  # pick every other row so that x-axis ticks are less busy
  newdf <- newdf[c(rep(c(TRUE, FALSE), (nrow(newdf)/2))), ]

  the_thm <- theme(plot.margin = unit(c(t=0 , r=0 , b=0, l=0), "cm"), 
                   panel.grid.major = element_line(size=0.2),
                   panel.spacing=unit(.5, "cm"),
                   legend.text=element_text(size=18, face="bold"),
                   legend.title = element_blank(),
                   legend.position = "bottom",
                   legend.margin=margin(t=-0.1, unit='cm'),
                   strip.text = element_text(face="bold", size=16, color="black"),
                   axis.text = element_text(size=13, color="black"), # face="bold",
                   axis.text.x = element_text(angle = 90),
                   axis.ticks = element_line(color = "black", size = .2),
                   axis.title.x = element_text(size=18, face="bold", margin=margin(t=2, r=0, b=0, l=0)),
                   axis.title.y = element_text(size=18, face="bold", margin=margin(t=0, r=10, b=0, l=0)),
                   plot.title = element_text(lineheight=.8, face="bold", size=20)
                   )

  ggplot(d1, aes(x=chill_dayofyear, y=value, fill=factor(variable))) +
  labs(x = "", y = "cumulative quantities", fill = "data type") +
  guides(fill=guide_legend(title="")) + 
  facet_grid(. ~ time_period ~ location, scales="fixed") +
  # geom_line(aes(fill=factor(Timeframe), color=factor(Timeframe) )) +
  stat_summary(geom="ribbon", alpha=0.2,
               fun =function(z) { quantile(z,0.5) }, 
               fun.min=function(z) { quantile(z,0)}, 
               fun.max=function(z) { quantile(z,1) }) +

  stat_summary(geom="ribbon", alpha=0.4,
               fun=function(z) { quantile(z,0.5) }, 
               fun.min=function(z) { quantile(z,0.1) }, 
               fun.max=function(z) { quantile(z,0.9) }) +

  stat_summary(geom="ribbon", alpha=0.8,
               fun=function(z) { quantile(z,0.5) }, 
               fun.min=function(z) { quantile(z,0.25) }, 
               fun.max=function(z) { quantile(z,0.75) }) + 

  stat_summary(geom="line", fun=function(z) {quantile(z,0.5)}) +

  scale_color_manual(values=c("darkgreen", "orange"),
                     breaks=c("cume_portions", "vert_Cum_dd"),
                     labels=c("CP accum.", "heat accum."))+
  
  scale_fill_manual(values=c("darkgreen", "orange"),
                    breaks=c("cume_portions", "vert_Cum_dd"),
                    labels=c("CP accum.", "heat accum.")) +
  
  # scale_y_continuous(breaks = xbreaks, label = xbreaks) +
  scale_x_continuous(breaks = newdf$day_count_since_sept, 
                     labels = newdf$letter_day) + 
  geom_hline(aes(yintercept = full_CP_intcpt),      size=0.2, color="darkgreen", alpha=0.3) + 
  # geom_hline(aes(yintercept = full_CP_intcpt*0.75), size=0.2, color="darkgreen", alpha=0.3, linetype="dashed") + 
  # geom_hline(aes(yintercept = heat_intcpt),         size=0.2, color="orange",    alpha=0.3) + 
  the_thm + 
  coord_cartesian(xlim = c(xmin, xmax), ylim = c(ymin, ymax)) #
  # theme_bw() + 
  # geom_text(data = slopes_dt,
  #           aes(label = sprintf(slopes_dt$str_slope), 
  #               x = rep((xmin + 25), length(slopes_dt$str_slope)),
  #               y = rep(220,         length(slopes_dt$str_slope))),
  #           colour = "black", fontface = "bold",
  #           size=5, 
  #           inherit.aes = FALSE
  #  )
}

double_cloud_2_rows_Accum_VertDD_CP_NoSlopes_200TrySpringerSub2_YMax <- function(d1, full_CP_intcpt=73.3, 
                                                                                 heat_intcpt=25, yMax=400){
  #
  # set the followings in a way that x-axis and y-axis have
  # the same length so that the slopes makes sense visually
  #
  ymin <- 0
  ymax <- yMax

  xmin <- 1
  xmax <- 185 # 250
   
  newdf <- data.table(chill_doy_map)
  # pick every other row so that x-axis ticks are less busy
  newdf <- newdf[c(rep(c(TRUE, FALSE), (nrow(newdf)/2))), ]

  the_thm <- theme(plot.margin = unit(c(t=0 , r=0 , b=0, l=0), "cm"), 
                   panel.grid.major = element_line(size=0.2),
                   panel.spacing=unit(.5, "cm"),
                   legend.text=element_text(size=18, face="bold"),
                   legend.title = element_blank(),
                   legend.position = "bottom",
                   legend.margin=margin(t=-0.1, unit='cm'),
                   strip.text = element_text(face="bold", size=16, color="black"),
                   axis.text = element_text(size=13, color="black"), # face="bold",
                   axis.text.x = element_text(angle = 90),
                   axis.ticks = element_line(color = "black", size = .2),
                   axis.title.x = element_text(size=18, face="bold", margin=margin(t=2, r=0, b=0, l=0)),
                   axis.title.y = element_text(size=18, face="bold", margin=margin(t=0, r=10, b=0, l=0)),
                   plot.title = element_text(lineheight=.8, face="bold", size=20)
                   )

  ggplot(d1, aes(x=chill_dayofyear, y=value, fill=factor(variable))) +
  labs(x = "", y = "cumulative quantities", fill = "data type") +
  guides(fill=guide_legend(title="")) + 
  facet_grid(. ~ time_period ~ location, scales="fixed") +
  # geom_line(aes(fill=factor(Timeframe), color=factor(Timeframe) )) +
  stat_summary(geom="ribbon", alpha=0.2,
               fun =function(z) { quantile(z,0.5) }, 
               fun.min=function(z) { quantile(z,0)}, 
               fun.max=function(z) { quantile(z,1) }) +

  stat_summary(geom="ribbon", alpha=0.4,
               fun=function(z) { quantile(z,0.5) }, 
               fun.min=function(z) { quantile(z,0.1) }, 
               fun.max=function(z) { quantile(z,0.9) }) +

  stat_summary(geom="ribbon", alpha=0.8,
               fun=function(z) { quantile(z,0.5) }, 
               fun.min=function(z) { quantile(z,0.25) }, 
               fun.max=function(z) { quantile(z,0.75) }) + 

  stat_summary(geom="line", fun=function(z) {quantile(z,0.5)}) +

  scale_color_manual(values=c("darkgreen", "orange"),
                     breaks=c("cume_portions", "vert_Cum_dd"),
                     labels=c("CP accum.", "heat accum."))+
  
  scale_fill_manual(values=c("darkgreen", "orange"),
                    breaks=c("cume_portions", "vert_Cum_dd"),
                    labels=c("CP accum.", "heat accum.")) +
  
  # scale_y_continuous(breaks = xbreaks, label = xbreaks) +
  scale_x_continuous(breaks = newdf$day_count_since_sept, 
                     labels = newdf$letter_day) + 
  # geom_hline(aes(yintercept = full_CP_intcpt), size=0.2, color="darkgreen", alpha=0.3) + 
  # geom_hline(aes(yintercept = full_CP_intcpt*0.75), size=0.2, color="darkgreen", alpha=0.3, linetype="dashed") + 
  # geom_hline(aes(yintercept = heat_intcpt),         size=0.2, color="orange",    alpha=0.3) + 
  the_thm + 
  coord_cartesian(xlim = c(xmin, xmax), ylim = c(ymin, ymax)) #
  # theme_bw() + 
  # geom_text(data = slopes_dt,
  #           aes(label = sprintf(slopes_dt$str_slope), 
  #               x = rep((xmin + 25), length(slopes_dt$str_slope)),
  #               y = rep(220,         length(slopes_dt$str_slope))),
  #           colour = "black", fontface = "bold",
  #           size=5, 
  #           inherit.aes = FALSE
  #  )
}

double_cloud_2_rows_Accum_VertDD_CP_NoSlopes_200TrySpringerSub2 <- function(d1, full_CP_intcpt=73.3, heat_intcpt=25){
  
  #
  # set the followings in a way that x-axis and y-axis have
  # the same length so that the slopes makes sense visually
  #
  ymin <- 0
  ymax <- 800

  xmin <- 1
  xmax <- 250
  # xmax <- 
   
  newdf <- data.table(chill_doy_map)
  # pick every other row so that x-axis ticks are less busy
  newdf <- newdf[c(rep(c(TRUE, FALSE), (nrow(newdf)/2))), ]

  the_thm <- theme(plot.margin = unit(c(t=0 , r=0 , b=0, l=0), "cm"), 
                   panel.grid.major = element_line(size=0.2),
                   panel.spacing=unit(.5, "cm"),
                   legend.text=element_text(size=18, face="bold"),
                   legend.title = element_blank(),
                   legend.position = "bottom",
                   legend.margin=margin(t=-0.1, unit='cm'),
                   strip.text = element_text(face="bold", size=16, color="black"),
                   axis.text = element_text(size=13, color="black"), # face="bold",
                   axis.text.x = element_text(angle = 90),
                   axis.ticks = element_line(color = "black", size = .2),
                   axis.title.x = element_text(size=18, face="bold", margin=margin(t=2, r=0, b=0, l=0)),
                   axis.title.y = element_text(size=18, face="bold", margin=margin(t=0, r=10, b=0, l=0)),
                   plot.title = element_text(lineheight=.8, face="bold", size=20)
                   )

  ggplot(d1, aes(x=chill_dayofyear, y=value, fill=factor(variable))) +
  labs(x = "", y = "cumulative quantities", fill = "data type") +
  guides(fill=guide_legend(title="")) + 
  facet_grid(. ~ time_period ~ location, scales="fixed") +
  # geom_line(aes(fill=factor(Timeframe), color=factor(Timeframe) )) +
  stat_summary(geom="ribbon", alpha=0.2,
               fun =function(z) { quantile(z,0.5) }, 
               fun.min=function(z) { quantile(z,0)}, 
               fun.max=function(z) { quantile(z,1) }) +

  stat_summary(geom="ribbon", alpha=0.4,
               fun=function(z) { quantile(z,0.5) }, 
               fun.min=function(z) { quantile(z,0.1) }, 
               fun.max=function(z) { quantile(z,0.9) }) +

  stat_summary(geom="ribbon", alpha=0.8,
               fun=function(z) { quantile(z,0.5) }, 
               fun.min=function(z) { quantile(z,0.25) }, 
               fun.max=function(z) { quantile(z,0.75) }) + 

  stat_summary(geom="line", fun=function(z) {quantile(z,0.5)}) +

  scale_color_manual(values=c("darkgreen", "orange"),
                     breaks=c("cume_portions", "vert_Cum_dd"),
                     labels=c("CP accum.", "heat accum."))+
  
  scale_fill_manual(values=c("darkgreen", "orange"),
                    breaks=c("cume_portions", "vert_Cum_dd"),
                    labels=c("CP accum.", "heat accum.")) +
  
  # scale_y_continuous(breaks = xbreaks, label = xbreaks) +
  scale_x_continuous(breaks = newdf$day_count_since_sept, 
                     labels = newdf$letter_day) + 
  geom_hline(aes(yintercept = full_CP_intcpt),      size=0.2, color="darkgreen", alpha=0.3) + 
  # geom_hline(aes(yintercept = full_CP_intcpt*0.75), size=0.2, color="darkgreen", alpha=0.3, linetype="dashed") + 
  # geom_hline(aes(yintercept = heat_intcpt),         size=0.2, color="orange",    alpha=0.3) + 
  the_thm + 
  coord_cartesian(xlim = c(xmin, xmax), ylim = c(ymin, ymax)) #
  # theme_bw() + 
  # geom_text(data = slopes_dt,
  #           aes(label = sprintf(slopes_dt$str_slope), 
  #               x = rep((xmin + 25), length(slopes_dt$str_slope)),
  #               y = rep(220,         length(slopes_dt$str_slope))),
  #           colour = "black", fontface = "bold",
  #           size=5, 
  #           inherit.aes = FALSE
  #  )
}



double_cloud_2_rows_Accum_VertDD_CP_Slopes_100Try <- function(d1, slopes_dt, full_CP_intcpt=73.3, heat_intcpt=25){
  
  #
  # set the followings in a way that x-axis and y-axis have
  # the same length so that the slopes makes sense visually
  #
  ymin <- 0
  ymax <- 220

  xmin <- 95
  xmax <- ymax + xmin - 1
  xmax <- 250
   
  newdf <- data.table(chill_doy_map)
  # pick every other row so that x-axis ticks are less busy
  newdf <- newdf[c(rep(c(TRUE, FALSE), (nrow(newdf)/2))), ]

  the_thm <- theme(plot.margin = unit(c(t=0 , r=0 , b=0, l=0), "cm"), 
                   panel.grid.major = element_line(size=0.2),
                   panel.spacing=unit(.5, "cm"),
                   legend.text=element_text(size=18, face="bold"),
                   legend.title = element_blank(),
                   legend.position = "bottom",
                   legend.margin=margin(t=-0.1, unit='cm'),
                   strip.text = element_text(face="bold", size=16, color="black"),
                   axis.text = element_text(size=16, color="black"), # face="bold",
                   axis.text.x = element_text(angle = 90),
                   axis.ticks = element_line(color = "black", size = .2),
                   axis.title.x = element_text(size=18, face="bold", margin=margin(t=2, r=0, b=0, l=0)),
                   axis.title.y = element_text(size=18, face="bold", margin=margin(t=0, r=10, b=0, l=0)),
                   plot.title = element_text(lineheight=.8, face="bold", size=20)
                   )

  ggplot(d1, aes(x=chill_dayofyear, y=value, fill=factor(variable))) +
  labs(x = "", y = "cumulative quantities", fill = "data type") +
  guides(fill=guide_legend(title="")) + 
  facet_grid(. ~ time_period ~ location, scales="fixed") +
  # geom_line(aes(fill=factor(Timeframe), color=factor(Timeframe) )) +
  stat_summary(geom="ribbon", alpha=0.2,
               fun =function(z) { quantile(z,0.5) }, 
               fun.min=function(z) { quantile(z,0)}, 
               fun.max=function(z) { quantile(z,1) }) +

  stat_summary(geom="ribbon", alpha=0.4,
               fun=function(z) { quantile(z,0.5) }, 
               fun.min=function(z) { quantile(z,0.1) }, 
               fun.max=function(z) { quantile(z,0.9) }) +

  stat_summary(geom="ribbon", alpha=0.8,
               fun=function(z) { quantile(z,0.5) }, 
               fun.min=function(z) { quantile(z,0.25) }, 
               fun.max=function(z) { quantile(z,0.75) }) + 

  stat_summary(geom="line", fun=function(z) {quantile(z,0.5)}) +

  scale_color_manual(values=c("darkgreen", "orange"),
                     breaks=c("cume_portions", "vert_Cum_dd"),
                     labels=c("CP accum.", "heat accum."))+
  
  scale_fill_manual(values=c("darkgreen", "orange"),
                    breaks=c("cume_portions", "vert_Cum_dd"),
                    labels=c("CP accum.", "heat accum.")) +
  
  # scale_y_continuous(breaks = xbreaks, label = xbreaks) +
  scale_x_continuous(breaks = newdf$day_count_since_sept, 
                     labels = newdf$letter_day) + 
  geom_hline(aes(yintercept = full_CP_intcpt),      size=0.2, color="darkgreen", alpha=0.3) + 
  geom_hline(aes(yintercept = full_CP_intcpt*0.75), size=0.2, color="darkgreen", alpha=0.3, linetype="dashed") + 
  geom_hline(aes(yintercept = heat_intcpt),         size=0.2, color="orange",    alpha=0.3) + 
  the_thm + # theme_bw() + 
  geom_text(data = slopes_dt,
            aes(label = sprintf(slopes_dt$str_slope), 
                x = rep((xmin + 25), length(slopes_dt$str_slope)),
                y = rep(220,         length(slopes_dt$str_slope))),
            colour = "black", fontface = "bold",
            size=5, 
            inherit.aes = FALSE
            ) + 
  coord_cartesian(xlim = c(xmin, xmax), ylim = c(ymin, ymax)) # 

}


box_and_LineKirti <- function(CP_quans, vertDD_quants){

  ymin <- 0
  ymax <- 650
  xmin <- 1
  xmax <- 250

  y_breaks <- unique(CP_quans$time_period)
   
  newdf <- data.table(chill_doy_map)
  # pick every other row so that x-axis ticks are less busy
  newdf <- newdf[c(rep(c(TRUE, FALSE), (nrow(newdf)/2))), ]

  ggplot() + 
  geom_linerange(data = CP_quans, 
                 aes(y = time_period, 
                     x = NULL, 
                     xmin = CP_quan_25,
                     xmax = CP_quan_75,
                     linetype = "CP accum."),
                 color = "darkgreen", 
                 size = 4,
                 show.legend = FALSE
                 ) +

  geom_linerange(data = vertDD_quants, 
                 aes(y = time_period, 
                     x = NULL, 
                     xmin = vertDD_quan_25,
                     xmax = vertDD_quan_1,
                     linetype = "heat accum."), 
                 color = "orange", 
                 size = 10,
                 alpha=0.6,
                 show.legend = FALSE
                 ) +
  labs(x = "", y = "", fill = "data type") +
  guides(fill=guide_legend(title="")) + 
  facet_grid(. ~ location, scales="fixed") +
  # geom_line(aes(fill=factor(Timeframe), color=factor(Timeframe) )) +
  scale_y_discrete(limits = levels(y_breaks)) + #expand=c(0.2, .1),
  scale_x_continuous(breaks = newdf$day_count_since_sept, 
                     labels = newdf$letter_day) + 

  theme(plot.margin = unit(c(t=0 , r=0.4 , b=0, l=-0.2), "cm"), 
        panel.grid.major = element_line(size=0.2),
        panel.spacing=unit(.5, "cm"),
        legend.text=element_text(size=18, face="bold"),
        legend.title = element_blank(),
        legend.position = "bottom",
        strip.text = element_text(face="bold", size=16, color="black"),
        axis.text = element_text(size=16, color="black"), # face="bold",
        axis.text.x = element_text(angle = 90),
        axis.ticks = element_line(color = "black", size = .2),
        axis.title.x = element_text(size=18,  face="bold", 
                                    margin=margin(t=10, r=0, b=0, l=0)),
        axis.title.y = element_text(size=18, face="bold",
                                    margin=margin(t=0, r=0, b=0, l=0)),
        plot.title = element_text(lineheight=.8, face="bold", size=20)
        ) + # theme_bw() + 
  coord_cartesian(xlim = c(95, 210)) # , ylim = c(0, 250))

}


double_cloud_2_rows_Accum_VertDD_CP_Springer_QuantLines <- function(d1, CP_quans, vertDD_quants){

  # xlabels <- sort(unique(d1$chill_dayofyear))
  # ylow = min(d1$value) - 5
  # ymax = min(260, max(d1$value))
  ymin <- 0
  ymax <- 650
  xmin <- 1
  xmax <- 250
   
  newdf <- data.table(chill_doy_map)
  # pick every other row so that x-axis ticks are less busy
  newdf <- newdf[c(rep(c(TRUE, FALSE), (nrow(newdf)/2))), ]

  ggplot(d1, aes(x=chill_dayofyear, y=value, fill=factor(variable))) +
  labs(x = "DoY in chill calendar", y = "cumulative quantities", fill = "data type") +
  guides(fill=guide_legend(title="")) + 
  facet_grid(. ~ time_period ~ location, scales="fixed") +
  # geom_line(aes(fill=factor(Timeframe), color=factor(Timeframe) )) +
  stat_summary(geom="ribbon", alpha=0.2,
               fun =function(z) { quantile(z,0.5) }, 
               fun.min=function(z) { quantile(z,0)}, 
               fun.max=function(z) { quantile(z,1) }) +

  stat_summary(geom="ribbon", alpha=0.4,
               fun=function(z) { quantile(z,0.5) }, 
               fun.min=function(z) { quantile(z,0.1) }, 
               fun.max=function(z) { quantile(z,0.9) }) +

  stat_summary(geom="ribbon", alpha=0.8,
               fun=function(z) { quantile(z,0.5) }, 
               fun.min=function(z) { quantile(z,0.25) }, 
               fun.max=function(z) { quantile(z,0.75) }) + 

  stat_summary(geom="line", fun=function(z) {quantile(z,0.5)}) +

  scale_color_manual(values=c("darkgreen", "orange"),
                     breaks=c("cume_portions", "vert_Cum_dd"),
                     labels=c("CP accum.", "heat accum."))+
  
  scale_fill_manual(values=c("darkgreen", "orange"),
                    breaks=c("cume_portions", "vert_Cum_dd"),
                    labels=c("CP accum.", "heat accum.")) +
  
  # scale_y_continuous(breaks = xbreaks, label = xbreaks) +
  scale_x_continuous(breaks = newdf$day_count_since_sept, 
                     labels = newdf$letter_day) + 
  
  # stat_summary(aes(colour = variable), geom = "hpline", width = 0.6, size = 1.5) + 
  geom_linerange(data = CP_quans, 
                 aes(y = y_intercept, 
                     x = NULL, 
                     xmin = CP_quan_25,
                     xmax = CP_quan_75), 
                 color = "red", 
                 size = 1.5,
                 show.legend = FALSE
                 ) +

  geom_linerange(data = vertDD_quants, 
                 aes(y = y_intercept, 
                     x = NULL, 
                     xmin = vertDD_quan_25,
                     xmax = vertDD_quan_1), 
                 color = "red", 
                 size = 1.5,
                 show.legend = FALSE
                 ) +

  # This removes data beyod the limits, use coord_cartesian(xlim = c(-5000, 5000)) 
  # limits = c(ylow, ymax)

  theme(plot.margin = unit(c(t=0 , r=0 , b=0, l=0), "cm"), 
        panel.grid.major = element_line(size=0.2),
        panel.spacing=unit(.5, "cm"),
        legend.text=element_text(size=18, face="bold"),
        legend.title = element_blank(),
        legend.position = "bottom",
        strip.text = element_text(face="bold", size=16, color="black"),
        axis.text = element_text(size=16, color="black"), # face="bold",
        axis.text.x = element_text(angle = 90),
        axis.ticks = element_line(color = "black", size = .2),
        axis.title.x = element_text(size=18,  face="bold", 
                                    margin=margin(t=10, r=0, b=0, l=0)),
        axis.title.y = element_text(size=18, face="bold",
                                    margin=margin(t=0, r=10, b=0, l=0)),
        plot.title = element_text(lineheight=.8, face="bold", size=20)
        ) + # theme_bw() + 
  coord_cartesian(xlim = c(120, 240), ylim = c(0, 250))

}


double_cloud_2_rows_Accum_VertDD_CP_Springer <- function(d1, y_interceptt=73.3){

  # xlabels <- sort(unique(d1$chill_dayofyear))
  # ylow = min(d1$value) - 5
  # ymax = min(260, max(d1$value))
  ymin <- 0
  ymax <- 650
  xmin <- 1
  xmax <- 250
   
  newdf <- data.table(chill_doy_map)
  # pick every other row so that x-axis ticks are less busy
  newdf <- newdf[c(rep(c(TRUE, FALSE), (nrow(newdf)/2))), ]

  ggplot(d1, aes(x=chill_dayofyear, y=value, fill=factor(variable))) +
  labs(x = "", y = "cumulative quantities", fill = "data type") +
  guides(fill=guide_legend(title="")) + 
  facet_grid(. ~ time_period ~ location, scales="fixed") +
  # geom_line(aes(fill=factor(Timeframe), color=factor(Timeframe) )) +
  stat_summary(geom="ribbon", alpha=0.2,
               fun =function(z) { quantile(z,0.5) }, 
               fun.min=function(z) { quantile(z,0)}, 
               fun.max=function(z) { quantile(z,1) }) +

  stat_summary(geom="ribbon", alpha=0.4,
               fun=function(z) { quantile(z,0.5) }, 
               fun.min=function(z) { quantile(z,0.1) }, 
               fun.max=function(z) { quantile(z,0.9) }) +

  stat_summary(geom="ribbon", alpha=0.8,
               fun=function(z) { quantile(z,0.5) }, 
               fun.min=function(z) { quantile(z,0.25) }, 
               fun.max=function(z) { quantile(z,0.75) }) + 

  stat_summary(geom="line", fun=function(z) {quantile(z,0.5)}) +

  scale_color_manual(values=c("darkgreen", "orange"),
                     breaks=c("cume_portions", "vert_Cum_dd"),
                     labels=c("CP accum.", "heat accum."))+
  
  scale_fill_manual(values=c("darkgreen", "orange"),
                    breaks=c("cume_portions", "vert_Cum_dd"),
                    labels=c("CP accum.", "heat accum.")) +
  
  # scale_y_continuous(breaks = xbreaks, label = xbreaks) +
  scale_x_continuous(breaks = newdf$day_count_since_sept, 
                     labels = newdf$letter_day) + 
  geom_hline(aes(yintercept = y_interceptt), color="red", alpha=0.3) + 
  # limits = c(ylow, ymax) This removes data beyod the limits, use coord_cartesian(xlim = c(-5000, 5000)) 
  
  theme(plot.margin = unit(c(t=0 , r=0 , b=0, l=0), "cm"), 
        panel.grid.major = element_line(size=0.2),
        panel.spacing=unit(.5, "cm"),
        legend.text=element_text(size=18, face="bold"),
        legend.title = element_blank(),
        legend.position = "bottom",
        strip.text = element_text(face="bold", size=16, color="black"),
        axis.text = element_text(size=16, color="black"), # face="bold",
        axis.text.x = element_text(angle = 90),
        axis.ticks = element_line(color = "black", size = .2),
        axis.title.x = element_text(size=18,  face="bold", 
                                    margin=margin(t=10, r=0, b=0, l=0)),
        axis.title.y = element_text(size=18, face="bold",
                                    margin=margin(t=0, r=10, b=0, l=0)),
        plot.title = element_text(lineheight=.8, face="bold", size=20)
        ) + # theme_bw() + 
  coord_cartesian(xlim = c(120, 240), ylim = c(0, 250))

}


ThreshCloud_2_rows_forForcing <- function(d1, trigger_dt){
  
  ##"""
  ##
  ## This function is modification of double_cloud_2_rows(.) to 
  ## plot threshold with shadowy background for Springer. Tossing the bloom
  ##
  ##"""
  
  d1$chill_season <- gsub("chill_", "", d1$chill_season)
  xlabels <- sort(unique(d1$chill_season))
  xlabels <- xlabels[seq(1, length(xlabels), 15)]
  xlabels <- c(xlabels) # , "2097-2098"

  d1$chill_season <- substr(d1$chill_season, 1, 4)
  d1$chill_season <- as.numeric(d1$chill_season)
  xbreaks <- sort(unique(d1$chill_season))
  xbreaks <- xbreaks[seq(1, length(xbreaks), 15)]
  # xbreaks <- c(xbreaks, 2097)
  # xbreaks <- c(xbreaks, 2100)
  ylow = min(d1$value) - 5
  ylow = 90

  ymax = 220 # this is tuned by visual inspection!!
  # ymax = min(ymax, max(d1$value))
  
  ggplot(d1, aes(x=chill_season, y=value, fill=factor(variable))) +
  labs(x = "chill year", y = "day of year", fill = "data type") +
  # guides(fill=guide_legend(title="")) + 

  facet_wrap(. ~ location, scales="fixed") +
  # geom_line(aes(fill=factor(Timeframe), color=factor(Timeframe) )) +
  stat_summary(geom="ribbon", fun =function(z) { quantile(z,0.5) }, 
                              fun.min=function(z) { quantile(z,0) }, 
                              fun.max=function(z) { quantile(z,1) }, 
               alpha=0.2) +

  stat_summary(geom="ribbon", fun=function(z) { quantile(z,0.5) }, 
                              fun.min=function(z) { quantile(z,0.1) }, 
                              fun.max=function(z) { quantile(z,0.9) }, 
               alpha=0.4) +

  stat_summary(geom="ribbon", fun=function(z) { quantile(z,0.5) }, 
                              fun.min=function(z) { quantile(z,0.25) }, 
                              fun.max=function(z) { quantile(z,0.75) }, 
               alpha=0.8) + 

  stat_summary(geom = "line", fun=function(z) {quantile(z,0.5)}) +
  # 
  # plot horizontal lines for each city.
  #
  geom_hline(data = trigger_dt, 
             aes(yintercept = median_heatTriggerThresh, 
                 linetype = "heat accumulation trigger"), 
             color = "red", 
             size=1,
             show.legend = TRUE
             ) + 
  scale_linetype_manual(name = "Limits", 
                        labels = c("heat accumulation trigger"), 
                        values = c("heat accumulation trigger" = 1)) +
  
  scale_color_manual(values=c("darkgreen"), breaks=c("thresh"), labels=c("CP threshold")) +
  scale_fill_manual (values=c("darkgreen"), breaks=c("thresh"), labels=c("CP threshold")) +
  
  scale_x_continuous(breaks = xbreaks, label = xbreaks) +
  scale_y_continuous(breaks = chill_doy_map$day_count_since_sept, 
                     labels = chill_doy_map$letter_day) + 

  theme(panel.grid.major = element_line(size=0.2),
        panel.spacing = unit(.5, "cm"),
        legend.text = element_text(size=18, face="bold"),
        legend.title = element_blank(),
        legend.position = "bottom",
        strip.text = element_text(face="bold", size=16, color="black"),
        axis.text = element_text(size=16, color="black"), # face="bold",
        # axis.text.x = element_text(angle=20, hjust = 1),
        axis.ticks = element_line(color = "black", size = .2),
        axis.title.x = element_text(size=18,  face="bold", 
                                    margin=margin(t=10, r=0, b=0, l=0)),
        axis.title.y = element_text(size=18, face="bold",
                                    margin=margin(t=0, r=10, b=0, l=0)),
        plot.title = element_text(lineheight=.8, face="bold", size=20)
        ) + # theme_bw() + 
  coord_cartesian(ylim = c(ylow, ymax)) + 
   guides(colour = guide_legend(override.aes = list(linetype = 0)),
         fill = guide_legend(override.aes = list(linetype = 0)),
         shape = guide_legend(override.aes = list(linetype = 0)),
         linetype = guide_legend())

}

cloudy_frost <- function(d1, colname="chill_dayofyear", fil){
  if (colname=="fifty_perc_chill_DoY"){
     # bloom color
     cls <- "orange"
     } else if(colname == "chill_dayofyear"){
      # frost color
      cls <- "deepskyblue"
      } else { # colname == "thresh_DoY")
        # threshold color
        cls <- "darkgreen"
  }
  d1$chill_season <- gsub("chill_", "", d1$chill_season)
  xbreaks <- sort(unique(d1$chill_season))
  xbreaks <- xbreaks[seq(1, length(xbreaks), 10)]
  xbreaks <- c(xbreaks)

  ggplot(d1, aes(x=chill_season, y=get(colname), 
                 fill=fil, group=time_period)) +
  labs(x = "chill year", y = "day of year") + #, fill = "Climate Group"
  # guides(fill=guide_legend(title="Time period")) + 
  facet_grid(. ~ emission ~ location) + # scales = "free"
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
               size = 1) + 
  # scale_x_continuous(breaks=seq(1970, 2100, 10)) +
  scale_x_discrete(breaks = xbreaks) +
  scale_y_continuous(breaks = chill_doy_map$day_count_since_sept, 
                     labels = chill_doy_map$letter_day) +
  scale_color_manual(values = cls) +
  scale_fill_manual(values = cls) +
  theme(panel.grid.major = element_line(size=0.2),
        panel.spacing=unit(.5, "cm"),
        legend.text=element_text(size=18, face="bold"),
        legend.title = element_blank(),
        legend.position = "bottom",
        strip.text = element_text(face="bold", size=16, color="black"),
        axis.text = element_text(size=16, color="black"), # face="bold",
        axis.text.x = element_text(angle=20, hjust = 1),
        axis.ticks = element_line(color = "black", size = .2),
        axis.title.x = element_text(size=18,  face="bold", 
                                    margin=margin(t=10, r=0, b=0, l=0)),
        axis.title.y = element_text(size=18, face="bold",
                                    margin=margin(t=0, r=10, b=0, l=0)),
        plot.title = element_text(lineheight=.8, face="bold", size=20)
        )
}

double_cloud <- function(d1){
  d1$chill_season <- gsub("chill_", "", d1$chill_season)
  xlabels <- sort(unique(d1$chill_season))
  xlabels <- xlabels[seq(1, length(xlabels), 15)]
  xlabels <- c(xlabels) # , "2097-2098"

  d1$chill_season <- substr(d1$chill_season, 1, 4)
  d1$chill_season <- as.numeric(d1$chill_season)
  xbreaks <- sort(unique(d1$chill_season))
  xbreaks <- xbreaks[seq(1, length(xbreaks), 15)]
  # xbreaks <- c(xbreaks, 2100)
  # xbreaks <- c(xbreaks, 2097)
  ylow = min(d1$value) - 5
  ymax = min(260, max(d1$value))
  ymax = 260
  
  ggplot(d1, aes(x=chill_season, y=value, fill=factor(variable))) +
  labs(x = "chill year", y = "day of year", fill = "data type") +
  guides(fill=guide_legend(title="")) + 
  facet_grid(. ~ emission ~ location, scales="free") +
  # geom_line(aes(fill=factor(Timeframe), color=factor(Timeframe) )) +
  stat_summary(geom="ribbon", fun.y=function(z) { quantile(z,0.5) }, 
                              fun.ymin=function(z) { quantile(z,0) }, 
                              fun.ymax=function(z) { quantile(z,1) }, 
               alpha=0.2) +

  stat_summary(geom="ribbon", fun.y=function(z) { quantile(z,0.5) }, 
                              fun.ymin=function(z) { quantile(z,0.1) }, 
                              fun.ymax=function(z) { quantile(z,0.9) }, 
               alpha=0.4) +

  stat_summary(geom="ribbon", fun.y=function(z) { quantile(z,0.5) }, 
                              fun.ymin=function(z) { quantile(z,0.25) }, 
                              fun.ymax=function(z) { quantile(z,0.75) }, 
               alpha=0.8) + 

  stat_summary(geom="line", fun.y=function(z) {quantile(z,0.5)}) +

  scale_color_manual(values=c("darkgreen", "orange"),
                     breaks=c("thresh", "fifty_perc_DoY"),
                     labels=c("CP threshold", "bloom"))+
  
  scale_fill_manual(values=c("darkgreen", "orange"),
                    breaks=c("thresh", "fifty_perc_chill_DoY"),
                    labels=c("CP threshold", "bloom")) +
  
  scale_x_continuous(breaks = xbreaks, label = xbreaks) +
  scale_y_continuous(breaks = chill_doy_map$day_count_since_sept, 
                     labels = chill_doy_map$letter_day
                     ) + 
# limits = c(ylow, ymax) This removes data beyod 
# the limits, use coord_cartesian(xlim = c(-5000, 5000)) 
  
  theme(panel.grid.major = element_line(size=0.2),
        panel.spacing=unit(.5, "cm"),
        legend.text=element_text(size=18, face="bold"),
        legend.title = element_blank(),
        legend.position = "bottom",
        strip.text = element_text(face="bold", size=16, color="black"),
        axis.text = element_text(size=16, color="black"), # face="bold",
        # axis.text.x = element_text(angle=20, hjust = 1),
        axis.ticks = element_line(color = "black", size = .2),
        axis.title.x = element_text(size=18,  face="bold", 
                                    margin=margin(t=10, r=0, b=0, l=0)),
        axis.title.y = element_text(size=18, face="bold",
                                    margin=margin(t=0, r=10, b=0, l=0)),
        plot.title = element_text(lineheight=.8, face="bold", size=20)
        ) + 
  # theme_bw() + 
  coord_cartesian(ylim = c(ylow, ymax)) 

}
####################################################################################

double_cloud_2_rows <- function(d1){
  d1$chill_season <- gsub("chill_", "", d1$chill_season)
  xlabels <- sort(unique(d1$chill_season))
  xlabels <- xlabels[seq(1, length(xlabels), 15)]
  xlabels <- c(xlabels) # , "2097-2098"

  d1$chill_season <- substr(d1$chill_season, 1, 4)
  d1$chill_season <- as.numeric(d1$chill_season)
  xbreaks <- sort(unique(d1$chill_season))
  xbreaks <- xbreaks[seq(1, length(xbreaks), 15)]
  # xbreaks <- c(xbreaks, 2097)
  # xbreaks <- c(xbreaks, 2100)
  ylow = min(d1$value) - 5
  ymax = min(260, max(d1$value))
  # ylow = 90
  ymax = 260
  
  ggplot(d1, aes(x=chill_season, y=value, fill=factor(variable))) +
  labs(x = "chill year", y = "day of year", fill = "data type") +
  guides(fill=guide_legend(title="")) + 
  facet_wrap(. ~ location, scales="fixed") +
  # geom_line(aes(fill=factor(Timeframe), color=factor(Timeframe) )) +
  stat_summary(geom="ribbon", fun =function(z) { quantile(z,0.5) }, 
                              fun.min=function(z) { quantile(z,0) }, 
                              fun.max=function(z) { quantile(z,1) }, 
               alpha=0.2) +

  stat_summary(geom="ribbon", fun=function(z) { quantile(z,0.5) }, 
                              fun.min=function(z) { quantile(z,0.1) }, 
                              fun.max=function(z) { quantile(z,0.9) }, 
               alpha=0.4) +

  stat_summary(geom="ribbon", fun=function(z) { quantile(z,0.5) }, 
                              fun.min=function(z) { quantile(z,0.25) }, 
                              fun.max=function(z) { quantile(z,0.75) }, 
               alpha=0.8) + 

  stat_summary(geom="line", fun=function(z) {quantile(z,0.5)}) +

  scale_color_manual(values=c("darkgreen", "orange"),
                     breaks=c("thresh", "fifty_perc_DoY"),
                     labels=c("CP threshold", "bloom"))+
  
  scale_fill_manual(values=c("darkgreen", "orange"),
                    breaks=c("thresh", "fifty_perc_chill_DoY"),
                    labels=c("CP threshold", "bloom")) +
  
  scale_x_continuous(breaks = xbreaks, label = xbreaks) +
  scale_y_continuous(breaks = chill_doy_map$day_count_since_sept, 
                     labels = chill_doy_map$letter_day
                     ) + 
  # This removes data beyod the limits, use coord_cartesian(xlim = c(-5000, 5000)) 
  # limits = c(ylow, ymax) 
  
  theme(panel.grid.major = element_line(size=0.2),
        panel.spacing=unit(.5, "cm"),
        legend.text=element_text(size=18, face="bold"),
        legend.title = element_blank(),
        legend.position = "bottom",
        strip.text = element_text(face="bold", size=16, color="black"),
        axis.text = element_text(size=16, color="black"), # face="bold",
        # axis.text.x = element_text(angle=20, hjust = 1),
        axis.ticks = element_line(color = "black", size = .2),
        axis.title.x = element_text(size=18,  face="bold", 
                                    margin=margin(t=10, r=0, b=0, l=0)),
        axis.title.y = element_text(size=18, face="bold",
                                    margin=margin(t=0, r=10, b=0, l=0)),
        plot.title = element_text(lineheight=.8, face="bold", size=20)
        ) + # theme_bw() + 
  coord_cartesian(ylim = c(ylow, ymax)) 

}
####################################################################################


cloudy_frost_2_rows <- function(d1, colname="chill_dayofyear", fil){
  if (colname=="fifty_perc_chill_DoY"){
     cls <- "orange" # bloom color
     } else if(colname == "chill_dayofyear"){
      cls <- "deepskyblue" # frost color
      } else { # colname == "thresh_DoY")
        cls <- "darkgreen" # threshold color
  }
  d1$chill_season <- gsub("chill_", "", d1$chill_season)
  d1$chill_season <- substr(d1$chill_season, 1, 4)
  d1$chill_season <- as.numeric(d1$chill_season)
  
  xbreaks <- sort(unique(d1$chill_season))
  xbreaks <- xbreaks[seq(1, length(xbreaks), 15)]

  ggplot(d1, aes(x=chill_season, y=get(colname), 
                 fill=fil, group=time_period)) +
  labs(x = "chill year", y = "day of year") + #, fill = "Climate Group"
  # guides(fill=guide_legend(title="Time period")) + 
  facet_wrap(. ~ location) + # scales = "free"
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
               size = 1) + 
  # scale_x_continuous(breaks=seq(1970, 2100, 10)) +
  scale_x_continuous(breaks = xbreaks) +
  scale_y_continuous(breaks = chill_doy_map$day_count_since_sept, 
                     labels = chill_doy_map$letter_day) +
  scale_color_manual(values = cls) +
  scale_fill_manual(values = cls) +
  theme(panel.grid.major = element_line(size=0.2),
        panel.spacing=unit(.5, "cm"),
        legend.text=element_text(size=18, face="bold"),
        legend.title = element_blank(),
        legend.position = "bottom",
        strip.text = element_text(face="bold", size=16, color="black"),
        axis.text = element_text(size=16, color="black"), # face="bold",
        # axis.text.x = element_text(angle=20, hjust = 1),
        axis.ticks = element_line(color = "black", size = .2),
        axis.title.x = element_text(size=18,  face="bold", 
                                    margin=margin(t=10, r=0, b=0, l=0)),
        axis.title.y = element_text(size=18, face="bold",
                                    margin=margin(t=0, r=10, b=0, l=0)),
        plot.title = element_text(lineheight=.8, face="bold", size=20)
        ) 
}