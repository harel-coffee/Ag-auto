rm(list=ls())
library(data.table)
library(dplyr)
library(ggpubr)

input_dir = "/Users/hn/Documents/01_research_data/codling_moth_2021/02_Diap_Gens_Vertdd/"
plot_path = "/Users/hn/Documents/01_research_data/codling_moth_2021/03_plots_4_paper/"


#### Relative

plot_rel_diapause <- function(data){

  emission = unique(data$emission)

  data <- data[variable =="RelLarvaPop" | variable =="RelNonDiap"]
  data <- subset(data, select=c("time_period", "city", "CumulativeDDF", "variable", "value"))
  
  time_periods_L <- c("Historical", "2040s", "2060s", "2080s")
  data$time_period = factor(data$time_period, levels=time_periods_L, order=T)

  city_levels = c("East Oroville", "Wenatchee", "Quincy", "Wapato", "Richland")
  data$city <- factor(data$city, levels = city_levels, ordered = TRUE)

  data$variable <- factor(data$variable)

  pp = ggplot(data, aes(x=CumulativeDDF, y=value, color=variable, fill=factor(variable))) + 
       labs(x = "cumulative degree days (in F)", y = "relative population", color = "relative population") +
       geom_vline(xintercept=c(213, 1153, 2313, 3443, 4453), linetype="solid", color ="grey", size=.25) +
       geom_hline(yintercept=c(5, 10, 15, 20), linetype = "solid", color = "grey", size=.25) +
       annotate(geom = "text", x = 700,  y = 16, label = "Gen. 1", color = "black", angle=30, size = 5) +
       annotate(geom = "text", x = 1700, y = 14, label = "Gen. 2", color = "black", angle=30, size = 5) +
       annotate(geom = "text", x = 2900, y = 12, label = "Gen. 3", color = "black", angle=30, size = 5) +
       annotate(geom = "text", x = 3920, y = 10, label = "Gen. 4", color = "black", angle=30, size = 5) +
       facet_grid(. ~ time_period ~  city, scales = "free") +
       scale_fill_manual(labels = c("total population", "population escaping diapause"), 
                         values=c("steelblue4", "orange"), 
                         name = "relative population") +
       scale_color_manual(labels = c("total population", "population escaping diapause"), 
                          values=c("steelblue4", "orange"), 
                          guide = FALSE) +
       stat_summary(geom="ribbon", fun = function(z) { quantile(z,0.5) }, 
                                   fun.min = function(z) { 0 }, 
                                   fun.max = function(z) { quantile(z,0.9) }, alpha=0.7) +
       # scale_x_continuous(limits = c(0, 4500)) + 
       scale_y_continuous(limits = c(0, 20)) +
       theme(panel.grid.major = element_blank(),
             panel.grid.minor = element_blank(),
             panel.spacing = unit(.25, "cm"),
             legend.title = element_blank(),
             legend.spacing.x = unit(.2, 'cm'),
             legend.position = "bottom",
             legend.key.size = unit(.65, "cm"),
             axis.ticks = element_line(color = "black", size = .2),
             legend.text = element_text(size=20),
             strip.text = element_text(size=20, face="bold"),
             axis.text = element_text(face = "plain", size=16, color="black"),
             plot.title = element_text(size=22, face="bold"),
             axis.title.x = element_text(face="bold", size=20, margin=margin(t=10, r=0, b=0, l=0), color="black"),
             axis.title.y = element_text(face="bold", size=20, margin=margin(t=0, r=10, b=0, l=0), color="black")
              ) +
       ggtitle(label = emission)
  return(pp)
}


diapause_rel <- readRDS(paste0(input_dir, "diapause_rel.rds"))

diapause_rel_45 <- diapause_rel %>% filter(emission == "RCP 4.5") %>% data.table()
diapause_rel_85 <- diapause_rel %>% filter(emission == "RCP 8.5") %>% data.table()
diapause_rel_observed <- diapause_rel %>% filter(emission == "Observed") %>% data.table()

diapause_rel_observed$time_period <- "Historical"

diapause_rel_observed_45 <- diapause_rel_observed
diapause_rel_observed_85 <- diapause_rel_observed

diapause_rel_observed_45$emission <- "RCP 4.5"
diapause_rel_observed_85$emission <- "RCP 8.5"

diap_rel_45 <- rbind(diapause_rel_45, diapause_rel_observed_45)
diap_rel_85 <- rbind(diapause_rel_85, diapause_rel_observed_85)

rel_45 <- plot_rel_diapause(data = diap_rel_45)
rel_85 <- plot_rel_diapause(data = diap_rel_85)


ggsave("relDiap_45.png", rel_45, path=plot_path, width = 20, height = 8, unit="in", dpi = 600, device="png")
ggsave("relDiap_85.png", rel_85, path=plot_path, width = 20, height = 8, unit="in", dpi = 600, device="png")

rel <- ggpubr::ggarrange(plotlist = list(rel_45, rel_85),
                         ncol = 2, nrow = 1,
                         common.legend = TRUE, legend = "bottom")

ggsave("relDiap_45_85.png", rel, path=plot_path, width = 35, height = 8, unit="in", dpi = 600, device="png")

rel <- ggpubr::ggarrange(plotlist = list(rel_45, rel_85),
                         ncol = 1, nrow = 2,
                         common.legend = TRUE, legend = "bottom")
ggsave("relDiap_45_85onTop.png", rel, path= plot_path, width = 20, height = 16, unit="in", dpi = 600, device="png")

# # A <- annotate_figure(adult_emerge, 
# #                      top = text_grob("Julian day of adult emergence", color="black", face="bold", size=35)
# #                      # fig.lab = "Julian day of adult emergence", fig.lab.size = 35, fig.lab.face = "bold")
# #                      )
# #### Absolute
# # file_name_extension = "diapause_abs_rcp85.rds"
# # version = "rcp85"
# # plot_abs_diapause(input_dir, file_name_extension, version, plot_path)

# file_name_extension = "diapause_abs_rcp45.rds"
# version = "rcp45"
# plot_abs_diapause(input_dir, file_name_extension, version, plot_path)

# A <- annotate_figure(adult_emerge, 
#                      top = text_grob("Julian day of adult emergence", color="black", face="bold", size=35)
#                      # fig.lab = "Julian day of adult emergence", fig.lab.size = 35, fig.lab.face = "bold")
#                      )
