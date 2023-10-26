# """
# This driver will comupte the percentage diff between future CPs and observed CPs.
# and generate a map of colors for them.
# 
# """
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

options(digits=9)
options(digit=9)


core_path = "/Users/hn/Documents/00_GitHub/Ag/chilling/chill_core.R"
plot_core_path = "/Users/hn/Documents/00_GitHub/Ag/chilling/chill_plot_core.R"
source(core_path)
source(plot_core_path)


data_dir = "/Users/hn/Documents/01_research_data/chilling/01_data/02/"
param_dir <- "/Users/hn/Documents/00_GitHub/Ag/chilling/parameters/"

LocationGroups_NoMontana <- read.csv(paste0(param_dir, "LocationGroups_NoMontana.csv"), 
                                     header=T, sep=",", as.is=T)

remove_montana <- function(data_dt, LocationGroups_NoMontana){
  if (!("location" %in% colnames(data_dt))){
    data_dt$location <- paste0(data_dt$lat, "_", data_dt$long)
  }
  data_dt <- data_dt %>% filter(location %in% LocationGroups_NoMontana$location)
  return(data_dt)
}


sept_summary_comp <- readRDS(paste0(data_dir, "sept_summary_comp.rds")) %>%
                     data.table()

head(sept_summary_comp, 2)
dim(sept_summary_comp)

keep_cols <- c("location", "sum_A1", "model", "emission", "time_period", "chill_season", "year")

sept_summary_comp <- subset(sept_summary_comp, select=keep_cols)


#
#  Toss RCP 4.5
#
sept_summary_comp <- sept_summary_comp %>% filter(emission != "rcp45")

#
#  Keep limited Locs
#
limited_CTs <- read.csv(paste0(param_dir, "4_locations.csv"), as.is=T)
limited_CTs$location <- paste0(limited_CTs$lat, "_", limited_CTs$long)
limited_CTs <- within(limited_CTs, remove(lat, long))

sept_summary_comp <- sept_summary_comp %>% 
                     filter(location %in% limited_CTs$location) %>% 
                     data.table()

sept_summary_comp <- dplyr::left_join(x = sept_summary_comp, y = limited_CTs, by = "location")
sept_summary_comp <- within(sept_summary_comp, remove(location))

future_and_observed <- sept_summary_comp %>% 
                       filter(time_period %in% c("2026-2050", "2051-2075", "2076-2099", "1979-2015")) %>% 
                       data.table()

future_and_modeledHist <- sept_summary_comp %>% 
                          filter(time_period%in% c("1950-2005", "2026-2050", "2051-2075", "2076-2099")) %>% 
                          data.table()

####################################################################################
####################################################################################
#
# form a base line for historical time period to subtract from future.
#
observed_base <- future_and_observed %>% 
                 filter(time_period == "1979-2015") %>% 
                 data.table()

observed_base[, base_sum_A1 := median(sum_A1), by=list(city)]
observed_base <- within(observed_base, remove(model,emission, time_period,chill_season, year, sum_A1))
observed_base <- data.table(unique(observed_base))


modHist_base <- future_and_modeledHist %>% 
                filter(time_period == "1950-2005") %>% 
                data.table()

modHist_base[, base_sum_A1 := median(sum_A1), by=list(model, city)]
modHist_base <- within(modHist_base, remove(emission, time_period,chill_season, year, sum_A1))
modHist_base <- data.table(unique(modHist_base))

####################################################################################
####################################################################################

diff_from_observed    <- dplyr::left_join(x = future_and_observed,     y = observed_base, by = "city")
diff_from_modeledHist <- dplyr::left_join(x = future_and_modeledHist, y = modHist_base , by = c("model", "city"))

diff_from_observed$diffs    <- diff_from_observed$sum_A1 - diff_from_observed$base_sum_A1
diff_from_modeledHist$diffs <- diff_from_modeledHist$sum_A1 - diff_from_modeledHist$base_sum_A1

diff_from_observed <- data.table(diff_from_observed)
diff_from_modeledHist <- data.table(diff_from_modeledHist)

diff_from_observed[, diffs_modelMedians := median(diffs), by=list(city, year)]
diff_from_modeledHist[, diffs_modelMedians := median(diffs), by=list(city, year)]


diff_from_observed <- subset(diff_from_observed, select= c(city, year, diffs_modelMedians))
diff_from_observed <- data.table(unique(diff_from_observed))

diff_from_modeledHist <- subset(diff_from_modeledHist, select= c(city, year, diffs_modelMedians))
diff_from_modeledHist <- data.table(unique(diff_from_modeledHist))



########################
########################  plot
########################

diff_from_modeledHist <- diff_from_modeledHist %>%
                         filter(year >= 2026) %>%
                         data.table()

diff_from_observed <- diff_from_observed %>%
                      filter(year >= 2026) %>%
                      data.table()

setnames(diff_from_modeledHist, old=c("diffs_modelMedians"), new=c("diffs_modelMedians_fromModeledHist"))
setnames(diff_from_observed,    old=c("diffs_modelMedians"), new=c("diffs_modelMedians_fromObserved"))


all_diffs <- dplyr::left_join(x = diff_from_observed, y = diff_from_modeledHist , by = c("year", "city"))


setnames(all_diffs, old=c("diffs_modelMedians_fromObserved", "diffs_modelMedians_fromModeledHist"), 
                    new=c("diff. from observed base", "diff. from modeled hist. base"))

all_diffs_melt <- melt(all_diffs, id = c("city", "year"))


diffs_plot <- ggplot(all_diffs_melt, aes(x=year, y=value, fill=variable, group=variable)) +
              labs(x = "year", y = "diff. from hist") + #, fill = "Climate Group"
              facet_grid(. ~ city) + # scales = "free"
              geom_line(aes(color=factor(variable))) +
              theme(panel.grid.major = element_line(size=0.2),
              panel.spacing=unit(.5, "cm"),
              legend.text=element_text(size=16),
              legend.title = element_blank(),
              legend.position = "bottom",
              strip.text = element_text(face="bold", size=16, color="black"),
              axis.text = element_text(size=12, color="black"), # face="bold",
              axis.text.x = element_text(angle=0, hjust = 1),
              axis.ticks = element_line(color = "black", size = .2),
              axis.title.x = element_text(size=16,
                                          margin=margin(t=10, r=0, b=0, l=0)),
              axis.title.y = element_text(size=16, 
                                          margin=margin(t=0, r=10, b=0, l=0)),
              plot.title = element_text(lineheight=.8, face="bold", size=20)
              ) + 
              ggtitle(label= "Accum. CP by March 31 - differences from history")


plot_path = "/Users/hn/Documents/00_GitHub/Ag_papers/Chill_Paper/MajorRevision/figures_4_revision/"
ggsave("diffs_plot.png", diffs_plot, path=plot_path, 
        dpi=400, 
        width=10, height=5, unit="in")




