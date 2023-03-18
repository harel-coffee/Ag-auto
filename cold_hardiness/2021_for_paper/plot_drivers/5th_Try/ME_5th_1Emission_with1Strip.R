rm(list=ls())
library(data.table)
library(ggplot2)
library(dplyr)
library(ggpubr) # for ggarrange
options(digits=9)
options(digit=9)


################################################################################
########################################
########################################
########################################
hardiness_core <- "/Users/hn/Documents/00_GitHub/Ag/cold_hardiness/2021_for_paper/hardiness_core.R"
hardiness_plot_core <- "/Users/hn/Documents/00_GitHub/Ag/cold_hardiness/2021_for_paper/hardiness_plot_core.R"

source(hardiness_core)
source(hardiness_plot_core)

########################################
########
########   directories
########
########################################
param_dir <- "/Users/hn/Documents/00_GitHub/Ag/cold_hardiness/parameters/"

data_dir_main <- "/Users/hn/Documents/01_research_data/cold_hardiness/hossein_locations/daily_data/"
data_dir <- data_dir_main

output_dir <- "/Users/hn/Documents/01_research_data/cold_hardiness/hossein_locations/Markus_Figrue_3/5th_try/1_emissions1Strip/"
if (!dir.exists(output_dir)) {
   dir.create(output_dir, recursive = T)
}

####
#### Three locations
####
three_locs <- read.csv(paste0(param_dir, "three_locations_CH.csv"), as.is=TRUE)
three_locs <- na.omit(three_locs)
three_locs$location <- paste0(three_locs$lat, "_", three_locs$long)
three_locs$locModel <- paste0(three_locs$location, "_", three_locs$model)
three_locs <- within(three_locs, remove(lat, long))

three_locs$facet <- paste(three_locs$location, three_locs$model, three_locs$CDI.change , sep=", ")

#
# proper DoY labeling
#
CH_DoY <- data.table(read.csv(paste0(param_dir, "CH_DoY_map.csv"), as.is=TRUE))
toDelete <- seq(0, nrow(CH_DoY), 2)
CH_DoY <-  CH_DoY[-toDelete, ]

########################################

CH_dt <- data.table::fread(paste0(data_dir, "all_chardonnay.csv.gz"))
CH_dt$locModel <- paste0(CH_dt$location, "_", CH_dt$model)

CH_dt_hist <- CH_dt %>% 
              filter(location %in% three_locs$location) %>% 
              filter(year %in% 1980:2003) %>% 
              data.table()
CH_dt_hist$time_period <- "1980-2003"

CH_dt_hist_45 <- CH_dt_hist
CH_dt_hist_85 <- CH_dt_hist

CH_dt_hist_45$emission <- "RCP 4.5"
CH_dt_hist_85$emission <- "RCP 8.5"

CH_dt <- CH_dt %>% 
         filter(locModel %in% three_locs$locModel) %>% 
         data.table()

CH_dt_F1 <- CH_dt %>% filter(hardiness_year %in% 2025:2049)
CH_dt_F2 <- CH_dt %>% filter(hardiness_year %in% 2050:2074)
CH_dt_F3 <- CH_dt %>% filter(hardiness_year %in% 2075:2099)

CH_dt_F1$time_period <- "2025-2049"
CH_dt_F2$time_period <- "2050-2074"
CH_dt_F3$time_period <- "2075-2099"

CH_dt <- rbind(CH_dt_F1, CH_dt_F2, CH_dt_F3, CH_dt_hist_45, CH_dt_hist_85)
CH_dt <- dplyr::left_join(x=CH_dt, y=three_locs[, c("location", "facet")])

CH_dt[CH_dt$emission=="rcp45", "emission"] <- "RCP 4.5"
CH_dt[CH_dt$emission=="rcp85", "emission"] <- "RCP 8.5"

CH_dt$emission <- factor(CH_dt$emission, 
                         levels =c("RCP 8.5", "Observed", "RCP 4.5"), 
                         order=TRUE)

CH_dt_median <- CH_dt %>%
                group_by(location, emission, variety, CH_DoY, facet, time_period)%>% 
                summarise(median_tmax=median(t_max), median_tmin=median(t_min), median_HC = median(predicted_Hc)) %>%
                data.table()

line_size = 0.6
fig_W <- 15
fig_H <- 10

the_theme <- theme(panel.grid.major = element_line(size=0.2),
                   panel.spacing=unit(.5, "cm"),
                   legend.text=element_text(size=12, face="bold"),
                   legend.title=element_blank(),
                   legend.position = "bottom",
                   strip.text = element_text(face="plain", size=12, color="black"),
                   axis.text = element_text(size=10, color="black"), # face="bold",
                   axis.text.x = element_text(angle = -90),
                   # axis.text.x = element_text(angle=20, hjust = 1),
                   axis.ticks = element_line(color = "black", size = .2),
                   axis.title.x = element_blank(),
                   axis.title.y = element_text(size=16, face="plain"),
                   
                   # axis.title.y = element_text(size=12, face="bold",
                   #                             margin=margin(t=0, r=10, b=0, l=0)),
                   plot.title = element_text(lineheight=.8, face="bold", size=12)
                   )

########### F1
# CH_dt_F1 <- CH_dt %>% filter(time_period %in% c("1980-2003", "2025-2049"))
# CH_dt_F2 <- CH_dt %>% filter(time_period %in% c("1980-2003", "2050-2074"))
# CH_dt_F3 <- CH_dt %>% filter(time_period %in% c("1980-2003", "2075-2099"))
a_fng_loc = unique(CH_dt$location)[1]

for (a_fng_em in unique(CH_dt$emission)){
  for (a_fng_loc in unique(CH_dt$location)){
    CH_dt_F1 <- CH_dt %>% filter(time_period %in% c("2025-2049") & location==a_fng_loc & emission==a_fng_em)
    CH_dt_F2 <- CH_dt %>% filter(time_period %in% c("2050-2074") & location==a_fng_loc & emission==a_fng_em)
    CH_dt_F3 <- CH_dt %>% filter(time_period %in% c("2075-2099") & location==a_fng_loc & emission==a_fng_em)
    CH_dt_obs<- CH_dt %>% filter(time_period %in% c("1980-2003") & location==a_fng_loc & emission==a_fng_em)

    min_y_f1 <- min(min(CH_dt_obs$t_min), min(CH_dt_F1$t_min))
    min_y_f2 <- min(min(CH_dt_obs$t_min), min(CH_dt_F2$t_min))
    min_y_f3 <- min(min(CH_dt_obs$t_min), min(CH_dt_F3$t_min))

    max_y_f1 <- max(max(CH_dt_obs$t_max), max(CH_dt_F1$t_max))
    max_y_f2 <- max(max(CH_dt_obs$t_max), max(CH_dt_F2$t_max))
    max_y_f3 <- max(max(CH_dt_obs$t_max), max(CH_dt_F3$t_max))


    CH_dt_F1_med <- CH_dt_median %>% filter(time_period %in% c("2025-2049") & location==a_fng_loc & emission==a_fng_em)
    CH_dt_F2_med <- CH_dt_median %>% filter(time_period %in% c("2050-2074") & location==a_fng_loc & emission==a_fng_em)
    CH_dt_F3_med <- CH_dt_median %>% filter(time_period %in% c("2075-2099") & location==a_fng_loc & emission==a_fng_em)
    CH_dt_obs_med<- CH_dt_median %>% filter(time_period %in% c("1980-2003") & location==a_fng_loc & emission==a_fng_em)

    # geom_line(aes(x=CH_DoY, y=t_min,        color=factor(time_period)), size=line_size) + 

    CH_F1_plt <- ggplot() +
                 geom_line(data=CH_dt_obs,aes(x=CH_DoY, y=predicted_Hc, group=factor(hardiness_year), linetype=time_period), size=line_size) + 
                 geom_line(data=CH_dt_F1, aes(x=CH_DoY, y=predicted_Hc, color=factor(hardiness_year), linetype=time_period), size=line_size) + 
                 geom_point(data=CH_dt_obs_med, aes(x=CH_DoY, y=median_HC), colour="black", size=(4*line_size), shape=15) + # linetype = "twodash"
                 geom_point(data=CH_dt_F1_med,  aes(x=CH_DoY, y=median_HC), colour="red",   size=(4*line_size), shape=15) + 
                 scale_x_continuous(breaks = CH_DoY$day_count_since_sept, labels=CH_DoY$letter_day) + 
                 facet_grid(. ~ facet) +
                 labs(y="cold hardiness") +
                 guides(color=FALSE, shape=TRUE) +
                 the_theme 



    CH_F2_plt <- ggplot() +
                 geom_line(data=CH_dt_obs,aes(x=CH_DoY, y=predicted_Hc, group=factor(hardiness_year), linetype=time_period), size=line_size) + 
                 geom_line(data=CH_dt_F2, aes(x=CH_DoY, y=predicted_Hc, color=factor(hardiness_year), linetype=time_period), size=line_size) + 
                 geom_point(data=CH_dt_obs_med, aes(x=CH_DoY, y=median_HC), colour="black", size=(4*line_size), shape=15) + # linetype = "twodash"
                 geom_point(data=CH_dt_F2_med,  aes(x=CH_DoY, y=median_HC), colour="red",   size=(4*line_size), shape=15) + 
                 scale_x_continuous(breaks = CH_DoY$day_count_since_sept, labels=CH_DoY$letter_day) + 
                 facet_grid(. ~ facet) +
                 labs(y="cold hardiness") +
                 guides(color=FALSE, shape=TRUE) +
                 the_theme


    CH_F3_plt <- ggplot() +
                 geom_line(data=CH_dt_obs,aes(x=CH_DoY, y=predicted_Hc, group=factor(hardiness_year), linetype=time_period), size=line_size) + 
                 geom_line(data=CH_dt_F3, aes(x=CH_DoY, y=predicted_Hc, color=factor(hardiness_year), linetype=time_period), size=line_size) + 
                 geom_point(data=CH_dt_obs_med, aes(x=CH_DoY, y=median_HC), colour="black", size=(4*line_size), shape=15) + # linetype = "twodash"
                 geom_point(data=CH_dt_F3_med,  aes(x=CH_DoY, y=median_HC), colour="red",   size=(4*line_size), shape=15) + 
                 scale_x_continuous(breaks=CH_DoY$day_count_since_sept, labels=CH_DoY$letter_day) + 
                 facet_grid(. ~ facet) +
                 labs(y="cold hardiness") +
                 guides(color=FALSE, shape=TRUE) +
                 the_theme


    tmin_F1_plt <- ggplot() +
                   geom_line(data=CH_dt_obs,aes(x=CH_DoY, y=t_min, group=factor(hardiness_year), linetype=time_period), size=line_size) + 
                   geom_line(data=CH_dt_F1, aes(x=CH_DoY, y=t_min, color=factor(hardiness_year), linetype=time_period), size=line_size) + 
                   geom_point(data=CH_dt_obs_med, aes(x=CH_DoY, y=median_tmin), colour="black", size=(4*line_size), shape=15) + # linetype = "twodash"
                   geom_point(data=CH_dt_F1_med,  aes(x=CH_DoY, y=median_tmin), colour="red",   size=(4*line_size), shape=15) + 
                   scale_x_continuous(breaks = CH_DoY$day_count_since_sept, labels=CH_DoY$letter_day) + 
                   facet_grid(. ~ facet) +
                   labs(y=expression(t[min])) +
                   guides(color=FALSE, shape=TRUE) +
                   the_theme + 
                   coord_cartesian(xlim = NULL,
                                 ylim = c(min_y_f1, max_y_f1),
                                 expand = TRUE,
                                 default = FALSE,
                                 clip = "on")

    tmin_F2_plt <- ggplot() +
                   geom_line(data=CH_dt_obs,aes(x=CH_DoY, y=t_min, group=factor(hardiness_year), linetype=time_period), size=line_size) + 
                   geom_line(data=CH_dt_F2, aes(x=CH_DoY, y=t_min, color=factor(hardiness_year), linetype=time_period), size=line_size) + 
                   geom_point(data=CH_dt_obs_med, aes(x=CH_DoY, y=median_tmin), colour="black", size=(4*line_size), shape=15) + # linetype = "twodash"
                   geom_point(data=CH_dt_F2_med,  aes(x=CH_DoY, y=median_tmin), colour="red",   size=(4*line_size), shape=15) + 
                   scale_x_continuous(breaks = CH_DoY$day_count_since_sept, labels=CH_DoY$letter_day) + 
                   facet_grid(. ~ facet) +
                   labs(y=expression(t[min])) +
                   guides(color=FALSE, shape=TRUE) +
                   the_theme + 
                   coord_cartesian(xlim = NULL,
                                 ylim = c(min_y_f2, max_y_f2),
                                 expand = TRUE,
                                 default = FALSE,
                                 clip = "on")

    tmin_F3_plt <- ggplot() +
                   geom_line(data=CH_dt_obs,aes(x=CH_DoY, y=t_min, group=factor(hardiness_year), linetype=time_period), size=line_size) + 
                   geom_line(data=CH_dt_F3, aes(x=CH_DoY, y=t_min, color=factor(hardiness_year), linetype=time_period), size=line_size) + 
                   geom_point(data=CH_dt_obs_med, aes(x=CH_DoY, y=median_tmin), colour="black", size=(4*line_size), shape=15) + # linetype = "twodash"
                   geom_point(data=CH_dt_F3_med,  aes(x=CH_DoY, y=median_tmin), colour="red",   size=(4*line_size), shape=15) + 
                   scale_x_continuous(breaks = CH_DoY$day_count_since_sept, labels=CH_DoY$letter_day) + 
                   facet_grid(. ~ facet) +
                   labs(y=expression(t[min])) +
                   guides(color=FALSE, shape=TRUE) +
                   the_theme + 
                   coord_cartesian(xlim = NULL,
                                 ylim = c(min_y_f3, max_y_f3),
                                 expand = TRUE,
                                 default = FALSE,
                                 clip = "on")

    tmax_F1_plt <- ggplot() +
                   geom_line(data=CH_dt_obs,aes(x=CH_DoY, y=t_max, group=factor(hardiness_year), linetype=time_period), size=line_size) + 
                   geom_line(data=CH_dt_F1, aes(x=CH_DoY, y=t_max, color=factor(hardiness_year), linetype=time_period), size=line_size) + 
                   geom_point(data=CH_dt_obs_med, aes(x=CH_DoY, y=median_tmax), colour="black", size=(4*line_size), shape=15) + # linetype = "twodash"
                   geom_point(data=CH_dt_F1_med,  aes(x=CH_DoY, y=median_tmax), colour="red",   size=(4*line_size), shape=15) + 
                   scale_x_continuous(breaks = CH_DoY$day_count_since_sept, labels=CH_DoY$letter_day) + 
                   facet_grid(. ~ emission ~ facet) +
                   labs(y=expression(t[max])) +
                   guides(color=FALSE, shape=TRUE) +
                   the_theme + 
                   coord_cartesian(xlim = NULL,
                                 ylim = c(min_y_f1, max_y_f1),
                                 expand = TRUE,
                                 default = FALSE,
                                 clip = "on")

    tmax_F2_plt <- ggplot() +
                   geom_line(data=CH_dt_obs,aes(x=CH_DoY, y=t_max, group=factor(hardiness_year), linetype=time_period), size=line_size) + 
                   geom_line(data=CH_dt_F2, aes(x=CH_DoY, y=t_max, color=factor(hardiness_year), linetype=time_period), size=line_size) + 
                   geom_point(data=CH_dt_obs_med, aes(x=CH_DoY, y=median_tmax), colour="black", size=(4*line_size), shape=15) + # linetype = "twodash"
                   geom_point(data=CH_dt_F2_med,  aes(x=CH_DoY, y=median_tmax), colour="red",   size=(4*line_size), shape=15) + 
                   scale_x_continuous(breaks = CH_DoY$day_count_since_sept, labels=CH_DoY$letter_day) + 
                   facet_grid(. ~ emission ~ facet) +
                   labs(y=expression(t[max])) +
                   guides(color=FALSE, shape=TRUE) +
                   the_theme + 
                   coord_cartesian(xlim = NULL,
                                 ylim = c(min_y_f2, max_y_f2),
                                 expand = TRUE,
                                 default = FALSE,
                                 clip = "on")

    tmax_F3_plt <- ggplot() +
                   geom_line(data=CH_dt_obs,aes(x=CH_DoY, y=t_max, group=factor(hardiness_year), linetype=time_period), size=line_size) + 
                   geom_line(data=CH_dt_F3, aes(x=CH_DoY, y=t_max, color=factor(hardiness_year), linetype=time_period), size=line_size) + 
                   geom_point(data=CH_dt_obs_med, aes(x=CH_DoY, y=median_tmax), colour="black", size=(4*line_size), shape=15) + # linetype = "twodash"
                   geom_point(data=CH_dt_F3_med,  aes(x=CH_DoY, y=median_tmax), colour="red",   size=(4*line_size), shape=15) + 
                   scale_x_continuous(breaks = CH_DoY$day_count_since_sept, labels=CH_DoY$letter_day) + 
                   facet_grid(. ~ emission ~ facet) +
                   labs(y=expression(t[max])) +
                   guides(color=FALSE, shape=TRUE) +
                   the_theme + 
                   coord_cartesian(xlim = NULL,
                                 ylim = c(min_y_f3, max_y_f3),
                                 expand = TRUE,
                                 default = FALSE,
                                 clip = "on")


    F1_plt <- ggarrange(plotlist = list(CH_F1_plt,
                                        tmin_F1_plt,
                                        tmax_F1_plt),
                        ncol = 3, nrow = 1,
                        legend = "bottom",
                        common.legend = TRUE)

    F2_plt <- ggarrange(plotlist = list(CH_F2_plt,
                                        tmin_F2_plt,
                                        tmax_F2_plt),
                        ncol = 3, nrow = 1,
                        legend = "bottom",
                        common.legend = TRUE)

    F3_plt <- ggarrange(plotlist = list(CH_F3_plt,
                                        tmin_F3_plt,
                                        tmax_F3_plt),
                        ncol = 3, nrow = 1,
                        legend = "bottom",
                        common.legend = TRUE)


    out_name_1 = paste0(a_fng_loc, "_", a_fng_em, "_", "2025-2049", ".pdf")
    out_name_2 = paste0(a_fng_loc, "_", a_fng_em, "_", "2050-2074", ".pdf")
    out_name_3 = paste0(a_fng_loc, "_", a_fng_em, "_", "2075-2099", ".pdf")
    
    ggsave(filename=out_name_1,
           plot=F1_plt, 
           path=output_dir, 
           width=fig_W, height=fig_H, unit="in", dpi=400, device = "pdf",
           limitsize = FALSE)

    ggsave(filename=out_name_2,
           plot=F2_plt, 
           path=output_dir, 
           width=fig_W, height=fig_H, unit="in", dpi=400, device = "pdf",
           limitsize = FALSE)

    ggsave(filename=out_name_3,
           plot=F3_plt, 
           path=output_dir, 
           width=fig_W, height=fig_H, unit="in", dpi=400, device = "pdf",
           limitsize = FALSE)
  }

}