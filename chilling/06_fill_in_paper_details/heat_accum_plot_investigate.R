# We could/should create two sets of data for each 
# (of the first two) 
# scenarios above or, we can take care of NAs
# -introduced to data by merging frost and bloom-
# in the plotting functions?
#####################################
rm(list=ls())
.libPaths("/data/hydro/R_libs35")
.libPaths()
library(data.table)
library(dplyr)
library(ggpubr)

options(digits=9)
options(digit=9)
############################################################
###
###             local computer source
###
############################################################
source_dir <- "/Users/hn/Documents/00_GitHub/Ag/Bloom/"
param_dir <- paste0(source_dir, "parameters/")


in_dir <- "/Users/hn/Documents/01_research_data/chilling/frost_bloom/"
plot_base_dir <- "/Users/hn/Documents/00_GitHub/Ag_papers/Chill_Paper/figures/heat_accumulation/"

#############################################################
###
###              
###
#############################################################

source_1 <- paste0(source_dir, "bloom_core.R")
source_2 <- paste0(source_dir, "bloom_plot_core.R")
source(source_1)
source(source_2)
#############################################################
###
###               Read data off the disk
###
#############################################################
limited_locations <- read.csv(file = paste0(param_dir, "limited_locations.csv"), as.is=TRUE)

limited_locations$location <- paste0(limited_locations$lat, "_", limited_locations$long)
limited_locations <- within(limited_locations, remove(lat, long))

#############################################################

heat <-  data.table(readRDS(paste0(in_dir, "heat_accum_limit_cities.rds")))
heat$location <- paste0(heat$lat, "_", heat$long)
heat <- within(heat, remove(lat, long, tmax, tmin, cripps_pink, gala, red_deli))

heat <- dplyr::left_join(x = heat, y = limited_locations, by = "location")

ict <- c("Omak", "Yakima", "Walla Walla", "Eugene")
heat <- heat %>% filter(city %in% ict)
heat$city <- factor(heat$city, levels = ict, order=TRUE)

dd = 15
heat_jan <- heat %>% filter(month == 1 & day == dd ) %>% data.table()

heat_jan_observed <- heat_jan %>% filter(model =="observed") %>% data.table()
heat_jan_future <- heat_jan %>% filter(model !="observed") %>% data.table()
heat_jan_future$time_period <- "Projection"
heat_jan_observed$time_period <- "Historical"
heat_jan <- rbind(heat_jan_observed, heat_jan_future)

cloudy_heat_accum_2_rows <- function(d1, fil="Heat accumulation in Jan."){ # colname="vert_Cum_dd",
  cls <- "red"
  xbreaks <- sort(unique(d1$year))
  xbreaks <- xbreaks[seq(1, length(xbreaks), 15)]

  ggplot(d1, aes(x=year, y=vert_Cum_dd, 
                 fill=fil, group=time_period)) +
  labs(x = "year", y = "accumulated heat in Jan.") + 
  facet_wrap(. ~ city) + # scales = "free"
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
  scale_x_continuous(breaks = xbreaks) +
  scale_color_manual(values = cls) +
  scale_fill_manual(values = cls) +
  theme(panel.grid.major = element_line(size=0.2),
        panel.spacing=unit(.5, "cm"),
        legend.text=element_text(size=18, face="bold"),
        legend.title = element_blank(),
        legend.position = "bottom",
        strip.text = element_text(face="bold", size=16, color="black"),
        axis.text = element_text(size=16, color="black"), 
        axis.ticks = element_line(color = "black", size = .2),
        axis.title.x = element_text(size=18,  face="bold", 
                                    margin=margin(t=10, r=0, b=0, l=0)),
        axis.title.y = element_text(size=18, face="bold",
                                    margin=margin(t=0, r=10, b=0, l=0)),
        plot.title = element_text(lineheight=.8, face="bold", size=20)
        ) 
}


jan_heat_plt <- cloudy_heat_accum_2_rows(heat_jan)

ggsave(plot=jan_heat_plt,
       filename =paste0("jan", dd, "_heat_accum_RCP85.png"), 
       width=15, height=10, units = "in", 
       dpi=600, device = "png",
       path=plot_base_dir)


heat_jan_hist <- heat_jan %>% filter(model == "observed")
jan_heat_plt <- cloudy_heat_accum_2_rows(heat_jan_hist)

ggsave(plot=jan_heat_plt,
       filename = paste0("historical_jan", dd, "_heat_accum_RCP85.png"), 
       width=15, height=10, units = "in", 
       dpi=600, device = "png",
       path=plot_base_dir)


add_time_periods <- function(dt){
  dt$time_period <- 0L
  dt_observed <- dt %>% filter(model == "observed")
  dt_F <- dt %>% filter(model != "observed")
  
  time_periods <- c("1979-2016", 
                    "2006-2025", "2026-2050", 
                    "2051-2075", "2076-2099")
  
  dt_observed$time_period <- "Historical"

  dt_F$time_period[dt_F$year >= 2006 & dt_F$year <= 2025] <- time_periods[2]
  dt_F$time_period[dt_F$year >= 2026 & dt_F$year <= 2050] <- time_periods[3]
  dt_F$time_period[dt_F$year >= 2051 & dt_F$year <= 2075] <- time_periods[4]
  dt_F$time_period[dt_F$year >= 2076] <- time_periods[5]

  dt <- rbind(dt_observed, dt_F)

  dt <- dt %>% filter(time_period != "2006-2025") %>% data.table()
  return(dt)
}


heat_box_plt <- function(data, colname="vert_Cum_dd"){
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
              labs(x="", y="accumulated heat") +
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

heat_jan <-  add_time_periods(heat_jan)

heat_jan_box_plot <- heat_box_plt(heat_jan)

output_name = paste0("heat_jan_", dd, "box_plot.png")
box_width = 6
box_height = 5
ggsave(output_name, heat_jan_box_plot, path=plot_base_dir, width=box_width, height=box_height, unit="in", dpi=600)





