rm(list=ls())
library(data.table)
library(dplyr)
library(ggplot2)


####################################################################################
data_dir <- "/Users/hn/Documents/01_research_data/codling_moth_2021/03_sensitivities/WSS/"
out_dir <- "/Users/hn/Documents/01_research_data/codling_moth_2021/03_plots_4_paper/"
####################################################################################

stages = c("Larva", "Adult")
deadlines = c("Aug", "Nov")

for (deadline in deadlines){
  for (stage in stages){
    if (deadline == "Aug"){
      observe_Fname <- "LO_gens_Aug_combined_CMPOP.rds"
      modeled_Fname <- "LM_gens_Aug_combined_CMPOP.rds"
      y_label_dead = "Aug. 23"
      }
      else{
      observe_Fname <- "LO_gens_Nov_combined_CMPOP.rds"
      modeled_Fname <- "LM_gens_Nov_combined_CMPOP.rds"
      y_label_dead = "Nov. 5"
    }
    
    #
    #  read the data
    #
    LO_gens <- data.table(readRDS(paste0(data_dir, observe_Fname)))
    LM_gens <- data.table(readRDS(paste0(data_dir, modeled_Fname)))
    LO_gens$time_period <- "Historical"

    ###
    ### Historical data is done without any shift.
    ### In historical data there are different shifts in the shift column
    ### But that is not true. It only means it was ran with the shifts that future models ran.
    ### i.e. if you look at all the data for shift 0 and other shifts, they are indentical.
    ### This will establish the straight line across x-axis (shifts)


    #
    # Take care of the oberved data so they appear in both RCPs
    #
    LO_gens_RCP45 <- LO_gens
    LO_gens_RCP85 <- LO_gens

    LO_gens_RCP45$emission <- "RCP 4.5"
    LO_gens_RCP85$emission <- "RCP 8.5"

    data <- rbind(LM_gens, LO_gens_RCP45, LO_gens_RCP85)

    # 
    #     Order time_period, emission, and cities
    #
    # data <- LM_gens
    time_levels = c("Historical", "2040s", "2060s", "2080s")
    data$time_period <- factor(data$time_period, levels = time_levels, ordered = TRUE)
    data$emission <- factor(data$emission, levels = c("RCP 4.5", "RCP 8.5"), ordered = TRUE)

    city_levels = c("East Oroville", "Wenatchee", "Quincy", "Wapato", "Richland")
    data$city <- factor(data$city, levels = city_levels, ordered = TRUE)

    ##########################################################################################
    #
    #  subset needed columns
    #

    if (stage == "Larva"){
      needed_columns = c("city", "shift", "time_period", "model", "emission", "NumLarvaGens")
      
      data    <- subset(data, select = needed_columns)
      # LO_gens <- subset(LO_gens, select = needed_columns)
      
      data <-  data %>% 
               group_by(city, shift, time_period, emission) %>% 
               summarise(NumLarvaGens = median(NumLarvaGens)) %>% 
               data.table()

      # LO_gens <- LO_gens %>% 
      #            group_by(city, shift, time_period, emission) %>% 
      #            summarise(NumLarvaGens = median(NumLarvaGens)) %>% 
      #            data.table()

      data$shift <- as.numeric(data$shift)

      } else {
      needed_columns = c("city", "shift", "time_period", "model", "emission", "NumAdultGens")
      
      # 
      #  subset the data. Really do not need to do it!
      # 
      data    <- subset(data, select = needed_columns)
      LO_gens <- subset(LO_gens, select = needed_columns)
      
      #
      #  aggregate the data.
      #
      data <-  data %>% 
               group_by(city, shift, time_period, emission) %>% 
               summarise(NumLarvaGens = median(NumAdultGens)) %>% 
               data.table()

      # LO_gens <-  LO_gens %>% 
      #             group_by(city, shift, time_period, emission) %>% 
      #             summarise(NumLarvaGens = median(NumAdultGens)) %>% 
      #             data.table()

      data$shift <- as.numeric(data$shift)
    }

    #######
    #######   Transfer the  datatable into the form from 2018 
    #######   so we can borrow the  plot's codes from 2018. 
    #######   Before doing that, lets see if we can get away with 
    #######   what we already have
    #######
    
    #
    # Tried to use the following theme for making the saving size smaller!
    #
    # the_theme <- theme(panel.grid.major = element_line(size = 0.05),
    #                    panel.grid.minor = element_blank(),
    #                    panel.spacing=unit(.5, "cm"),
    #                    legend.title = element_blank(), # element_text(face = "plain", size = 16),
    #                    legend.text = element_text(size = 3),
    #                    legend.position = "bottom",
    #                    legend.key.size = unit(.3, "line"),
    #                    legend.margin = margin(t= -0.2, r=0, b=.2, l=0, unit = 'cm'),
    #                    strip.text = element_text(size = 3, face = "bold"),
    #                    axis.text = element_text(face="plain", size = 2, color="black"),
    #                    axis.ticks = element_line(color = "black", size = .01),
    #                    axis.title.x = element_text(face= "plain", size = 4, margin = margin(t = 1, r = 0, b = 0, l = 0)),
    #                    axis.title.y = element_text(face= "plain", size = 4, margin = margin(t = 0, r = 1, b = 0, l = 0)),
    #                    axis.ticks.length=unit(.04, "cm"))


    axis_text_size <- 8
    strip_text_size <- 8
    plot_title_font_size <- 10
    legent_font_size <- 10
    the_theme <- theme(plot.margin = unit(c(t = 0.2, r = 0.2, b = -0.2, l = 0.2), "cm"),
                   panel.border = element_rect(fill=NA, size=.3),
                   plot.title = element_text(size = plot_title_font_size, face = "bold"),
                   panel.grid.major = element_line(size = 0.05),
                   panel.grid.minor = element_blank(),
                   panel.spacing=unit(.25,"cm"),
                   legend.position="bottom", 
                   legend.title = element_blank(),
                   legend.key.size = unit(1, "line"),
                   legend.text = element_text(size = legent_font_size),
                   legend.margin = margin(t= -0.1, r=0, b=.5, l=0, unit = 'cm'),
                   legend.spacing.x = unit(.05, 'cm'),
                   strip.text.x = element_text(size = strip_text_size, face="bold"),
                   strip.text.y = element_text(size = strip_text_size, face="bold"),
                   axis.ticks = element_line(color = "black", size = .2),
                   # axis.ticks.x = element_blank(),
                   axis.text.x = element_text(size = axis_text_size, face = "plain", color = "black"),
                   axis.text.y = element_text(size = axis_text_size, face="plain", color = "black"),
                   axis.title.x = element_text(face = "bold", size = axis_text_size, margin = margin(t=6, r=0, b=0, l=0)),
                   axis.title.y = element_text(face = "bold", size = axis_text_size, margin = margin(t=0, r=6, b=0, l=0)),
                  )

    color_ord = c("grey47", "dodgerblue", "olivedrab4", "red")

    plott <- ggplot(data, aes(x=shift*100, y = NumLarvaGens, color = time_period)) + 
             geom_point(size = 0.1) +
             geom_line() +
             facet_grid( ~ emission ~ city, scales="free") + 
             scale_color_manual(values=color_ord,
                                name = "Time\nPeriod",
                                labels = c("Historical", "2040s", "2060s", "2080s")) +
             scale_fill_manual(values = color_ord,
                               name = "Time\nPeriod",
                               labels = c("Historical", "2040s", "2060s", "2080s")) + 
             the_theme + 
             labs(x = "Weibull scale parameter change by %", 
                  y = paste0("No. of ", tolower(stage), " generations (by ", y_label_dead, ")"))

    file_name <- tolower(paste0("no_", stage, "_gens_by_", gsub(". ", "", y_label_dead), ".png"))
    
    ggsave(filename = file_name, 
           path = out_dir, 
           plot = plott,
           width = 9, height = 5, units = "in",
           dpi = 400,
           device = "png")
  }
}





