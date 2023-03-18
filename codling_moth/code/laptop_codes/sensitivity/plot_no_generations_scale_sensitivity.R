rm(list=ls())
library(data.table)
library(dplyr)
library(ggplot2)


main_in <- "/Users/hn/Documents/GitHub/Kirti/codling_moth/to_write_paper/figures/sensitivity/weibull_scale/"
setwd(main_in)

numeric_shifts <- c(0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07,
                    0.08, 0.09, 0.1, 0.11, 0.12,
                    0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2)

all_info <- data.table()
files <- dir(pattern=".csv")

for (file in files){
  current_file <- data.table(read.csv(file, check.names=FALSE))
  current_file <- subset(current_file, select=c(1, 2))
  cc <- colnames(current_file)[2]
  current_file$emission <- paste0("RCP ", unlist(strsplit(cc, "_"))[1])
  current_file$stage <- unlist(strsplit(cc, "_"))[2]
  current_file$due <- unlist(strsplit(cc, "_"))[3]
  current_file$time_period <- unlist(strsplit(cc, "_"))[4]
  setnames(current_file, old= colnames(current_file)[2], new=c("no_gens"))
  current_file <- melt(current_file, id=c("shift", "emission", "stage", "due", "time_period"))
  all_info <- rbind(all_info, current_file)
}

all_info <- all_info %>% filter(time_period != "historical")

legend_labels = c("2040s", "2060s", "2080s")

rm(list=ls())
library(data.table)
library(dplyr)
library(ggplot2)

master_path <- "/Users/hn/Documents/GitHub/Kirti/codling_moth/to_write_paper/figures/sensitivity/weibull_scale/"
setwd(main_in)
numeric_shifts <- c(0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07,
                    0.08, 0.09, 0.1, 0.11, 0.12,
                    0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2)


file_pref = "rcp"
model_type = c("45", "85")
time_period = c ("_historical", "_2040", "_2060", "_2080")
weather_type = c("_warm", "_cold")
file_suffix = ".csv"

for (weather in weather_type){
  for (model in model_type){
    # initialize a data table, so we could use cbind
    all_info = data.table(numeric_shifts)
    for (time in time_period){
      file_name = paste0(file_pref, model, time, weather, file_suffix)
      file = paste0(master_path, file_name)
      current_file = data.table(read.csv(file, check.names=FALSE))
      current_file = within(current_file, remove(shift))
      if (time=="_historical"){
        current_file[, 1] = current_file[1, 1]
        current_file[, 2] = current_file[1, 2]
        current_file[, 3] = current_file[1, 3]
        current_file[, 4] = current_file[1, 4]
      }
      all_info <- cbind(all_info, current_file)
    }
    all_info = melt(all_info, id=c("numeric_shifts"))
    colnames(all_info) <- c("shifts", "pop_type", "generation")
  
    # plot the poulations
    dead_lines = c("Aug", "Nov")
    stages = c("_Larva_", "_Adult_")
    for (dead in dead_lines){
      for (stag in stages){
        if (dead == "Aug"){d = "Aug. 23"} else {d = "Nov. 5"}
        if (model== "45") {m = "RCP 4.5"} else {m = "RCP 8.5"}
        if (weather == "_warm") {aa = " (Warmer Areas)"} else {aa= " (Cooler Areas)"}
        plot_title <- paste0(m, aa)

        mask_entry = paste0(model, stag, dead, "_")
        mask = c(paste0(mask_entry, "2040"), paste0(mask_entry, "2060"),
                 paste0(mask_entry, "2080"))
      
        curr_data = all_info[all_info$pop_type %in% mask]
      
        legend_labels = factor(c("2040s", "2060s", "2080s"), order=T)
      
        h_line_coord = as.numeric(all_info[all_info$pop_type %in% c(paste0(mask_entry, "historical"))][1, 3])
        history_line <- data.frame( x = c(-Inf, Inf), y = h_line_coord, history_line = factor(h_line_coord) )
        dot_plot =ggplot(curr_data, aes(x=shifts*100, y=generation, color=pop_type)) + 
                  geom_point() +
                  geom_line() + 
                  geom_line(aes(x, y, linetype = "Historical" ), history_line, inherit.aes = FALSE) +
                  ylim(1, 3.25) + 
                  theme(panel.grid.major = element_line(size = 0.3),
                        panel.grid.minor = element_line(size = 0.2),
                        plot.margin = unit(c(t=0.2, r=.2, b=1, l=.2), "cm"),
                        panel.spacing=unit(.5, "cm"),
                        legend.position="bottom",
                        legend.title = element_blank(),
                        plot.title = element_text(size=25, face="bold"),
                        legend.text = element_text(size=25, face="bold"),
                        legend.spacing.x = unit(.05, 'cm'),
                        legend.key.size = unit(1, "cm"),
                        legend.margin=margin(t= -.5, r = 0, b = 0, l = 0),
                        axis.ticks = element_line(color = "black", size = .2),
                        axis.text = element_text(face = "bold", size=22),
                        axis.title.x = element_text(face = "bold", size=20, margin = margin(t=15, r=0, b=0, l=0)),
                        axis.title.y = element_text(face = "bold", size=20, margin = margin(t=0, r=15, b=0, l=0)),
                  ) + 
                  scale_color_discrete(breaks=mask, labels= legend_labels) +
                  labs(x="Weibull scale parameter change by %", y="No. of generations") +
                  ggtitle(label=plot_title)
                  plot_name = paste0(file_pref, model, stag, dead, "_scale_sens", weather)
                  assign(x = plot_name, value ={dot_plot})
                  # ggsave(paste0(plot_name, ".png"), dot_plot, path=master_path, device="png", 
                  #        dpi=500, width=5.57, height=5.42, unit="in")
      }
    }
  }
}

larva <- ggpubr::ggarrange(plotlist = list(rcp45_Larva_Aug_scale_sens_cold, rcp45_Larva_Aug_scale_sens_warm,
                                           rcp85_Larva_Aug_scale_sens_cold, rcp85_Larva_Aug_scale_sens_warm),
                           ncol = 2, nrow = 2,
                           common.legend = TRUE, legend = "bottom")



ggsave("larva_sensitivity.png", larva, path=master_path, device="png", 
       dpi=400, width=12, height=12, unit="in")
