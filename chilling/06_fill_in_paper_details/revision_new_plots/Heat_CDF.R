
rm(list = ls())
library(data.table)
library(dplyr)
library(lemon)

data_dir <- "/Users/hn/Documents/01_research_data/chilling/frost_bloom/"

source_dir <- "/Users/hn/Documents/00_GitHub/Ag/Bloom/"
param_dir <- paste0(source_dir, "parameters/")

################################################################################################

limited_locations <- read.csv(file = paste0(param_dir, "limited_locations.csv"), as.is=TRUE)

limited_locations$location <- paste0(limited_locations$lat, "_", limited_locations$long)
limited_locations <- within(limited_locations, remove(lat, long))


daily_heat_for_CDF <- readRDS(paste0(data_dir, "/heat_accum_limit_cities.rds"))
daily_heat_for_CDF$location <- paste0(daily_heat_for_CDF$lat, "_", daily_heat_for_CDF$long)

daily_heat_for_CDF <- subset(daily_heat_for_CDF, 
                             select=c("year", "month", "day", "model", "emission", "location", "vert_Cum_dd"))

daily_heat_for_CDF <- dplyr::left_join(x = daily_heat_for_CDF, 
                                       y = limited_locations, 
                                       by = "location")

daily_heat_for_CDF <- daily_heat_for_CDF %>%
                      filter(city %in% c("Omak", "Walla Walla", "Eugene", "Yakima")) %>%
                      data.table()

CTs <- c("Omak", "Yakima", "Walla Walla", "Eugene")
data$city <- factor(data$city, levels = CTs, order=TRUE)

daily_heat_for_CDF$newDay = 1

daily_heat_for_CDF <- daily_heat_for_CDF %>% 
                      group_by(year, emission, city, model ) %>% 
                      mutate(DoY = cumsum(newDay)) %>% 
                      data.table()


daily_heat_for_CDF_observed <- daily_heat_for_CDF %>% 
                               filter(model == "observed") %>%
                               data.table()

daily_heat_for_CDF_modeled <- daily_heat_for_CDF %>% 
                              filter(model != "observed" ) %>% 
                              data.table()


the_theme <- theme(panel.grid.major = element_line(size=0.2),
                   panel.spacing=unit(.5, "cm"),
                   legend.text=element_text(size=12, face="plain"),
                   legend.title = element_blank(),
                   legend.position = "none",
                   strip.text = element_text(face="plain", size=16, color="black"),
                   axis.text = element_text(face="bold", size=8, color="black"),
                   axis.ticks = element_line(color = "black", size = .2),
                   axis.title.x = element_text(face="plain", size=16, 
                                               margin=margin(t=10, r=0, b=0, l=0)),

                   axis.title.y = element_text(face="plain", size=16, 
                                               margin=margin(t=0, r=10, b=0, l=0)),

                   plot.title = element_text(lineheight=.8, face="bold")
                   )

a_plot <- ggplot(daily_heat_for_CDF_observed, aes(x=DoY, y=vert_Cum_dd, colour = year)) +
          facet_grid(. ~ emission ~ city, scales="free") + 
          geom_line(aes(color=factor(year) )) + 
          labs(x = "DoY", y = "cumulative vert. accum. DD", fill = "data type") +
          scale_x_continuous(breaks=c(  31,   59,    90,    120,    151,   181,  212,   243, 273,   304,   334, 365),
                             labels=c("J31", "F28", "M31", 'A30', "M31", "J30", "J31", "A31", "S30", "O31", "N30", "D31")
                             ) + 
          geom_vline(xintercept = 31, color = "black", size = 0.4) + 
          geom_vline(xintercept = 59, color = "black", size = 0.4) + 
          geom_vline(xintercept = 90, color = "black", size = 0.4) + 
          geom_vline(xintercept = 120, color = "black", size = 0.4) + 
          geom_vline(xintercept = 151, color = "black", size = 0.4) + 
          geom_vline(xintercept = 181, color = "black", size = 0.4) +
          geom_vline(xintercept = 212, color = "black", size = 0.4) +
          geom_vline(xintercept = 243, color = "black", size = 0.4) +
          the_theme

output_name = "observed_cumulative_AccumHeat.png"
plot_dir = "/Users/hn/Documents/00_GitHub/Ag_papers/Chill_Paper/figures_4_revision/"
box_width = 16
box_height = 4
ggsave(output_name, a_plot, path=plot_dir, 
       width=box_width, height=box_height, 
       unit="in", dpi=600)

daily_heat_for_CDF_modeled_F1 <- daily_heat_for_CDF_modeled %>%
                                filter(year <= 2050) %>% 
                                data.table()

daily_heat_for_CDF_modeled_F2 <- daily_heat_for_CDF_modeled %>%
                               filter(year > 2050) %>% 
                               data.table()

daily_heat_for_CDF_modeled_F2 <- daily_heat_for_CDF_modeled_F2 %>%
                               filter(year <= 2075) %>% 
                               data.table()


daily_heat_for_CDF_modeled_F3 <- daily_heat_for_CDF_modeled %>%
                               filter(year > 2075) %>% 
                               data.table()

plot_Future_Heat <- function(dt){
  the_theme <- theme(panel.grid.major = element_line(size=0.2),
                   panel.spacing=unit(.5, "cm"),
                   legend.text=element_text(size=24, face="plain"),
                   legend.title = element_blank(),
                   legend.position = "none",
                   strip.text = element_text(face="plain", size=28, color="black"),
                   axis.text = element_text(face="plain", size=16, color="black"),
                   axis.text.x = element_text(size=16, face="plain", 
                                              color="black", angle=-90),
                   axis.ticks = element_line(color = "black", size = .2),
                   axis.title.x = element_text(face="bold", size=32, 
                                               margin=margin(t=10, r=0, b=0, l=0)),
   
                   axis.title.y = element_text(face="plain", size=32, 
                                               margin=margin(t=0, r=10, b=0, l=0)),
   
                   plot.title = element_text(lineheight=.8, face="plain")
                   )
  ggplot(dt, aes(x=DoY, y=vert_Cum_dd, colour = year)) +
        facet_grid(. ~ emission ~ city ~ model, scales="fixed") + 
        facet_rep_grid(. ~ emission ~ city ~ model, scales="fixed",  repeat.tick.labels = TRUE) + 
        geom_line(aes(color=factor(year) )) + 
        labs(x = "DoY", y = "cumulative accumulated heat", fill = "data type") +
        scale_x_continuous(breaks=c(  31,   59,    90,    120,    151,   181,  212,   243, 273,   304,   334, 365),
                           labels=c("J31", "F28", "M31", 'A30', "M31", "J30", "J31", "A31", "S30", "O31", "N30", "D31")
                           ) +  
        geom_vline(xintercept = 31, color = "black", size = 0.4) + 
        geom_vline(xintercept = 59, color = "black", size = 0.4) + 
        geom_vline(xintercept = 90, color = "black", size = 0.4) + 
        geom_vline(xintercept = 120, color = "black", size = 0.4) + 
        geom_vline(xintercept = 151, color = "black", size = 0.4) + 
        geom_vline(xintercept = 181, color = "black", size = 0.4) +
        geom_vline(xintercept = 212, color = "black", size = 0.4) +
        geom_vline(xintercept = 243, color = "black", size = 0.4) +
        the_theme
}


box_width = 80
box_height = 35

a_plot <- plot_Future_Heat(daily_heat_for_CDF_modeled_F1)
output_name = "F1_modeled_cumulative_Accumheat.png"
ggsave(output_name, 
       a_plot, 
       path=plot_dir, 
       width=box_width, height=box_height, unit="in", dpi=200,
       limitsize = FALSE)

a_plot <- plot_Future_Heat(daily_heat_for_CDF_modeled_F2)
output_name = "F2_modeled_cumulative_Accumheat.png"
ggsave(output_name, 
       a_plot, 
       path=plot_dir, 
       width=box_width, height=box_height, unit="in", dpi=200,
       limitsize = FALSE)

a_plot <- plot_Future_Heat(daily_heat_for_CDF_modeled_F3)
output_name = "F3_modeled_cumulative_Accumheat.png"
ggsave(output_name, 
       a_plot, 
       path=plot_dir, 
       width=box_width, height=box_height, unit="in", dpi=200,
       limitsize = FALSE)




