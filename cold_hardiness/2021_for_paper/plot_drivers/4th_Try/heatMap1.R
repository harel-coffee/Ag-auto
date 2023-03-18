rm(list=ls())
library(data.table)
library(ggplot2)
library(dplyr)
library(viridis) # for scale_fill_viridis()
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

output_dir <- "/Users/hn/Documents/01_research_data/cold_hardiness/hossein_locations/Markus_Figrue_3/4th_try/"
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
              filter(model == "Observed") %>%  #  %in% 1980:2003
              data.table()

CH_dt_hist_45 <- CH_dt_hist
CH_dt_hist_85 <- CH_dt_hist

CH_dt_hist_45$emission <- "RCP 4.5"
CH_dt_hist_85$emission <- "RCP 8.5"

CH_dt <- CH_dt %>% 
         filter(locModel %in% three_locs$locModel) %>% 
         filter(model != "Observed") %>%  #  %in% 1980:2003
         filter(hardiness_year %in% 2025:2049) %>%
         data.table()

CH_dt <- rbind(CH_dt, CH_dt_hist_45, CH_dt_hist_85)
CH_dt <- dplyr::left_join(x=CH_dt, y=three_locs[, c("location", "facet")])

CH_dt[CH_dt$emission=="rcp45", "emission"] <- "RCP 4.5"
CH_dt[CH_dt$emission=="rcp85", "emission"] <- "RCP 8.5"

CH_dt$emission <- factor(CH_dt$emission, 
                         levels =c("RCP 8.5", "Observed", "RCP 4.5"), 
                         order=TRUE)

CH_dt <- CH_dt %>% filter(CH_DoY<=257)


the_theme <- theme(panel.grid.major = element_line(size=0.2),
                   panel.spacing=unit(.5, "cm"),
                   legend.text=element_text(size=10, face="plain"),
                   legend.title=element_text(size=14, face="plain"),
                   legend.position = "bottom",
                   strip.text = element_text(face="plain", size=12, color="black"),
                   axis.text = element_text(size=10, color="black"), # face="bold",
                   axis.text.x = element_text(angle = -90),
                   # axis.text.x = element_text(angle=20, hjust = 1),
                   axis.ticks = element_line(color = "black", size = .2),
                   axis.title.x = element_blank(),
                   axis.title.y = element_text(size=14, face="plain",
                                               margin=margin(t=0, r=10, b=0, l=0)),
                   plot.title = element_text(lineheight=.8, face="bold", size=12)
                   )

a_plt <- ggplot(CH_dt, aes(CH_DoY, hardiness_year, fill=predicted_Hc)) + 
         geom_tile() + 
         facet_grid(. ~ emission ~ facet, scale="free") + 
         scale_x_continuous(breaks=CH_DoY$day_count_since_sept, labels=CH_DoY$letter_day) +
         scale_y_continuous(breaks=seq(from = 1980, to = 2050, by = 10) ) + 
         ylab("year") + labs(fill="CH") + 
         the_theme + 
         scale_fill_viridis(discrete=FALSE) # scale_fill_gradient() or scale_fill_distiller() or scale_fill_viridis(discrete=FALSE)

fig_W <- 15
fig_H <- 10
ggsave(plot = a_plt,
       filename=paste0("4th_Try_Heat_", "scale_fill_viridis", ".pdf"),
       width=fig_W, height=fig_H, units = "in", 
       dpi=400, device = "pdf",
       limitsize = FALSE,
       path = output_dir)

