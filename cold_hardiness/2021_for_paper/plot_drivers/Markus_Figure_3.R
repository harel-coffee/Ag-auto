library(data.table)
library(ggplot2)
library(dplyr)
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

data_dir_main <- "/Users/hn/Documents/01_research_data/cold_hardiness/hossein_locations/daily_data/"

data_dir <- data_dir_main

output_dir <- "/Users/hn/Documents/01_research_data/cold_hardiness/hossein_locations/Markus_Figrue_3/"
if (!dir.exists(output_dir)) {
   dir.create(output_dir, recursive = T)
}

grape_varieties <- c("cabernet_sauvignon", "chardonnay", "merlot", "reisling")

########################################

# grape_varieties = grape_varieties[1]
for (a_variety in grape_varieties){
  CH_dt <- data.table::fread(paste0(data_dir, "all_", a_variety, ".csv.gz"))

  CH_dt_observed <- CH_dt %>%
                    filter(emission == "Observed") %>%
                    data.table()

  CH_dt_observed$time_period <- "Observed"

  CH_dt_F <- CH_dt %>%
             filter(emission != "Observed") %>%
             data.table()

  CH_dt_F <- CH_dt_F %>%
                  filter(year >= 2024) %>%
                  data.table()

  CH_dt_F <- CH_dt_F %>%
             mutate(time_period = case_when(hardiness_year %in% c(2026:2050) ~ "2026-2050",
                                            hardiness_year %in% c(2051:2075) ~ "2051-2075",
                                            hardiness_year %in% c(2076:2099) ~ "2076-2099")
                    )

  CH_dt_F <- CH_dt_F[!is.na(CH_dt_F$time_period),]

  CH_dt <- rbind(CH_dt_observed, CH_dt_F)
  rm(CH_dt_observed, CH_dt_F)





}



CH_dt_melt <- melt(CH_dt, id = c("location", "time_period", "variety", "emission"))


m_plt <- ggplot(CH_dt_melt) +
         labs(y = "predicted HC                          temperature") + 
         geom_line(aes(x = variable, y = value, size=1) + 
         facet_grid(. ~ emission ~ location)


          scale_color_manual(values = c('Cabernet Sauvignon' = 'darkgreen',
                                        'Chardonnay' = 'red',
                                        'Merlot' = 'orange',
                                        'Riesling' = 'slateblue1')) +
                             labs(color = 'grape variety') + 
          scale_x_continuous(breaks = CH_DoY$day_count_since_sept, labels= CH_DoY$letter_day) + 
          ggtitle(paste0(unique(weather$location), ", ", unique(CH_dt$emission), ", ", unique(CH_dt$time_period))) +
          theme(panel.grid.major = element_line(size=0.2),
                panel.spacing=unit(.5, "cm"),
                legend.text=element_text(size=12, face="bold"),
                legend.title = element_blank(),
                legend.position = "bottom",
                strip.text = element_text(face="bold", size=12, color="black"),
                axis.text = element_text(size=12, color="black"), # face="bold",
                # axis.text.x = element_text(angle=20, hjust = 1),
                axis.ticks = element_line(color = "black", size = .2),
                axis.title.x = element_blank(),
                axis.title.y = element_text(size=12, face="bold",
                                            margin=margin(t=0, r=10, b=0, l=0)),
                plot.title = element_text(lineheight=.8, face="bold", size=12)
                ) 




