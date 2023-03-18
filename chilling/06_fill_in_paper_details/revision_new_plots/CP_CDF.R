
rm(list = ls())
library(data.table)
library(dplyr)
library(lemon)

data_dir <- "/Users/hn/Documents/01_research_data/chilling/01_data/"
daily_CP_for_CDF <- read.csv(paste0(data_dir, "/daily_CP_for_CDF.csv"), as.is=TRUE)

CTs <- c("Omak", "Yakima", "Walla Walla", "Eugene")
daily_CP_for_CDF$city <- factor(daily_CP_for_CDF$city, levels = CTs, order=TRUE)

daily_CP_for_CDF$newDay = 1
daily_CP_for_CDF_A <- daily_CP_for_CDF %>% 
                      group_by(chill_season, emission, city, model ) %>% 
                      mutate(chillYearDoY = cumsum(newDay)) %>% 
                      data.table()


daily_CP_for_CDF_observed <- daily_CP_for_CDF_A %>% 
                             filter(emission=="observed") %>%
                             data.table()

daily_CP_for_CDF_modeled <- daily_CP_for_CDF_A %>% 
                            filter(emission %in% c("RCP 4.5", "RCP 8.5") ) %>% 
                            data.table()

daily_CP_for_CDF_observed_45 <- daily_CP_for_CDF_observed
daily_CP_for_CDF_observed_85 <- daily_CP_for_CDF_observed

daily_CP_for_CDF_observed_45$emission <- "RCP 4.5"
daily_CP_for_CDF_observed_85$emission <- "RCP 8.5"

data_4_plot <- rbind(daily_CP_for_CDF_observed_45, daily_CP_for_CDF_observed_85, daily_CP_for_CDF_modeled)


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

a_plot <- ggplot(daily_CP_for_CDF_observed, aes(x=chillYearDoY, y=cume_portions, colour = chill_season)) +
          facet_grid(. ~ emission ~ city, scales="free") + 
          geom_line(aes(color=factor(chill_season) )) + 
          labs(x = "chill DoY", y = "cumulative accumulated CP", fill = "data type") +
          scale_x_continuous(breaks=c(   30,   61,    91,   122,   153,   181,   212,   242),
                             labels=c("S30", "O31", "N30", 'D31', "J31", "F28", "M31", "A30")
                             ) + 
          geom_vline(xintercept = 30, color = "black", size = 0.4) + 
          geom_vline(xintercept = 61, color = "black", size = 0.4) + 
          geom_vline(xintercept = 91, color = "black", size = 0.4) + 
          geom_vline(xintercept = 122, color = "black", size = 0.4) + 
          geom_vline(xintercept = 153, color = "black", size = 0.4) + 
          geom_vline(xintercept = 181, color = "black", size = 0.4) +
          the_theme

output_name = "observed_cumulative_AccumCP.png"
plot_dir = "/Users/hn/Documents/00_GitHub/Ag_papers/Chill_Paper/figures_4_revision/"
box_width = 16
box_height = 4
ggsave(output_name, a_plot, path=plot_dir, 
       width=box_width, height=box_height, 
       unit="in", dpi=600)

daily_CP_for_CDF_modeled_F1 <- daily_CP_for_CDF_modeled %>%
                               filter(chill_season <= "chill_2049-2050") %>% 
                               data.table()

daily_CP_for_CDF_modeled_F2 <- daily_CP_for_CDF_modeled %>%
                               filter(chill_season > "chill_2049-2050") %>% 
                               data.table()

daily_CP_for_CDF_modeled_F2 <- daily_CP_for_CDF_modeled_F2 %>%
                               filter(chill_season <= "chill_2074-2075") %>% 
                               data.table()


daily_CP_for_CDF_modeled_F3 <- daily_CP_for_CDF_modeled %>%
                               filter(chill_season > "chill_2074-2075") %>% 
                               data.table()

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

a_plot <- ggplot(daily_CP_for_CDF_modeled_F1, aes(x=chillYearDoY, y=cume_portions, colour = chill_season)) +
          facet_grid(. ~ emission ~ city ~ model, scales="fixed") + 
          facet_rep_grid(. ~ emission ~ city ~ model, scales="fixed",  repeat.tick.labels = TRUE) + 
          geom_line(aes(color=factor(chill_season) )) + 
          labs(x = "chill DoY", y = "cumulative accumulated CP", fill = "data type") +
          scale_x_continuous(breaks=c(   30,   61,    91,   122,   153,   181,   212,   242),
                             labels=c("S30", "O31", "N30", 'D31', "J31", "F28", "M31", "A30")
                             ) + 
          geom_vline(xintercept = 30, color = "black", size = 0.4) + 
          geom_vline(xintercept = 61, color = "black", size = 0.4) + 
          geom_vline(xintercept = 91, color = "black", size = 0.4) + 
          geom_vline(xintercept = 122, color = "black", size = 0.4) + 
          geom_vline(xintercept = 153, color = "black", size = 0.4) + 
          geom_vline(xintercept = 181, color = "black", size = 0.4) +
          the_theme

output_name = "F1_modeled_cumulative_AccumCP.png"
plot_dir = "/Users/hn/Documents/00_GitHub/Ag_papers/Chill_Paper/figures_4_revision/"
box_width = 80
box_height = 35
ggsave(output_name, a_plot, 
       path=plot_dir, 
       width=box_width, height=box_height, unit="in", dpi=200,
       limitsize = FALSE)




a_plot <- ggplot(daily_CP_for_CDF_modeled_F2, aes(x=chillYearDoY, y=cume_portions, colour = chill_season)) +
          facet_grid(. ~ emission ~ city ~ model, scales="fixed") + 
          facet_rep_grid(. ~ emission ~ city ~ model, scales="fixed",  repeat.tick.labels = TRUE) + 
          geom_line(aes(color=factor(chill_season) )) + 
          labs(x = "chill DoY", y = "cumulative accumulated CP", fill = "data type") +
          scale_x_continuous(breaks=c(   30,   61,    91,   122,   153,   181,   212,   242),
                             labels=c("S30", "O31", "N30", 'D31', "J31", "F28", "M31", "A30")
                             ) + 
          geom_vline(xintercept = 30, color = "black", size = 0.4) + 
          geom_vline(xintercept = 61, color = "black", size = 0.4) + 
          geom_vline(xintercept = 91, color = "black", size = 0.4) + 
          geom_vline(xintercept = 122, color = "black", size = 0.4) + 
          geom_vline(xintercept = 153, color = "black", size = 0.4) + 
          geom_vline(xintercept = 181, color = "black", size = 0.4) +
          the_theme

output_name = "F2_modeled_cumulative_AccumCP.png"
plot_dir = "/Users/hn/Documents/00_GitHub/Ag_papers/Chill_Paper/figures_4_revision/"
box_width = 80
box_height = 35
ggsave(output_name, a_plot, 
       path=plot_dir, 
       width=box_width, height=box_height, unit="in", dpi=200,
       limitsize = FALSE)




a_plot <- ggplot(daily_CP_for_CDF_modeled_F3, aes(x=chillYearDoY, y=cume_portions, colour = chill_season)) +
          facet_grid(. ~ emission ~ city ~ model, scales="fixed") + 
          facet_rep_grid(. ~ emission ~ city ~ model, scales="fixed",  repeat.tick.labels = TRUE) + 
          geom_line(aes(color=factor(chill_season) )) + 
          labs(x = "chill DoY", y = "cumulative accumulated CP", fill = "data type") +
          scale_x_continuous(breaks=c(   30,   61,    91,   122,   153,   181,   212,   242),
                             labels=c("S30", "O31", "N30", 'D31', "J31", "F38", "M31", "A30")
                             ) + 
          geom_vline(xintercept = 30, color = "black", size = 0.4) + 
          geom_vline(xintercept = 61, color = "black", size = 0.4) + 
          geom_vline(xintercept = 91, color = "black", size = 0.4) + 
          geom_vline(xintercept = 122, color = "black", size = 0.4) + 
          geom_vline(xintercept = 153, color = "black", size = 0.4) + 
          geom_vline(xintercept = 181, color = "black", size = 0.4) +
          the_theme

output_name = "F3_modeled_cumulative_AccumCP.png"
plot_dir = "/Users/hn/Documents/00_GitHub/Ag_papers/Chill_Paper/figures_4_revision/"
box_width = 80
box_height = 35
ggsave(output_name, a_plot, 
       path=plot_dir, 
       width=box_width, height=box_height, unit="in", dpi=200,
       limitsize = FALSE)




