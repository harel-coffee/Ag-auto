rm(list=ls())
library(chron)
library(data.table)
library(reshape2)
library(dplyr)
library(foreach)
library(iterators)
library(ggplot2)

plot_abs_diapause <- function(input_dir, file_name_extension, 
                              version, plot_path, ii, max_y){
  ##
  ## input_dir 
  ## file_name_extension, 
  ## version either rcp45 or rcp85
  ## plot_path 
  ## 
  file_name = paste0(input_dir, file_name_extension)
  data <- data.table(readRDS(file_name))
  data$CountyGroup = as.character(data$CountyGroup)
  data[CountyGroup == 1]$CountyGroup = 'Cooler Areas'
  data[CountyGroup == 2]$CountyGroup = 'Warmer Areas'
  data <- data[variable =="AbsLarvaPop" | variable =="AbsNonDiap"]
  data <- subset(data, select=c("ClimateGroup", "CountyGroup", "CumulativeDDF", "variable", "value"))
  data$variable <- factor(data$variable)
  diap_plot <-  ggplot(data, aes(x=CumulativeDDF, y=value, color=variable, fill=factor(variable))) + 
                labs(x = "Cumulative degree days (in F)", y = "Absolute population", color = "Absolute population") +
                geom_vline(xintercept=c(213, 1153, 2313, 3443, 4453), linetype="solid", color ="grey", size=.25) +
                geom_hline(yintercept=seq(0, max_y, 25), linetype="solid", color ="grey", size=.25) +
                annotate(geom="text", x=700,  y=85, label="Gen. 1", color="black", angle=30) +
                annotate(geom="text", x=1700, y=80, label="Gen. 2", color="black", angle=30) + 
                annotate(geom="text", x=2900, y=75, label="Gen. 3", color="black", angle=30) + 
                annotate(geom="text", x=3920, y=70, label="Gen. 4", color="black", angle=30) +
                theme_bw() +
                facet_grid(. ~ CountyGroup ~ ClimateGroup, scales = "free") +
                theme(axis.text = element_text(face= "plain", size = 8, color="black"),
                      panel.grid.major = element_blank(),
                      panel.grid.minor = element_blank(),
                      axis.title.x = element_text(face= "plain", size = 12, 
                                                  margin = margin(t=10, r = 0, b = 0, l = 0)),
                      axis.title.y = element_text(face= "plain", size = 12, 
                                                  margin = margin(t=0, r = 10, b = 0, l = 0)),
                      legend.position="bottom"
                      ) + 
                scale_fill_manual(labels = c("Total", "Escape diapause"), values=c("grey", "orange"), name = "Absolute population") +
                scale_color_manual(labels = c("Total", "Escape diapause"), values=c("grey", "orange"), guide = FALSE) +
                stat_summary(geom="ribbon", fun.y=function(z) { quantile(z,0.5) }, 
                             fun.ymin=function(z) { 0 }, 
                             fun.ymax=function(z) { quantile(z,0.9) }, alpha=0.7) +
                scale_x_continuous(limits = c(0, 5000)) + 
                scale_y_continuous(limits = c(0, max_y), breaks=seq(0, max_y, 25))
              
  plot_name = paste0("diapause_abs_", version, "_", ii, ".png")
  ggsave(plot_name, diap_plot, device="png", 
  	     path=plot_path, width=10, height=7, 
         unit="in", dpi=400)
}

plot_rel_diapause <- function(input_dir, file_name_extension, 
                              version, plot_path, ii){
  file_name = paste0(input_dir, file_name_extension)
  data <- data.table(readRDS(file_name))
  data$CountyGroup = as.character(data$CountyGroup)
  data[CountyGroup == 1]$CountyGroup = 'Cooler Areas'
  data[CountyGroup == 2]$CountyGroup = 'Warmer Areas'
  data <- data[variable =="RelLarvaPop" | variable =="RelNonDiap"]
  data <- subset(data, select=c("ClimateGroup", "CountyGroup", "CumulativeDDF", "variable", "value"))
  data$variable <- factor(data$variable)
  
  pp = ggplot(data, aes(x=CumulativeDDF, y=value, color=variable, fill=factor(variable))) + 
       labs(x = "Cumulative degree days(in F)", y = "Relative population", color = "Relative population") +
       geom_vline(xintercept=c(213, 1153, 2313, 3443, 4453), linetype="solid", color ="grey", size=.25) +
       geom_hline(yintercept=c(5, 10, 15, 20), linetype="solid", color ="grey", size=.25) +
       annotate(geom="text", x=700,  y=18, label="Gen. 1", color="black", angle=30) +
       annotate(geom="text", x=1700, y=16, label="Gen. 2", color="black", angle=30) + 
       annotate(geom="text", x=2900, y=14, label="Gen. 3", color="black", angle=30) + 
       annotate(geom="text", x=3920, y=12, label="Gen. 4", color="black", angle=30) +
       theme_bw() +
       facet_grid(. ~ CountyGroup ~ ClimateGroup, scales = "free") +
       theme(axis.text = element_text(face= "plain", size = 8, color="black"),
             panel.grid.major = element_blank(),
             panel.grid.minor = element_blank(),
             axis.title.x = element_text(face= "plain", size = 12, margin = margin(t=10, r = 0, b = 0, l = 0)),
             axis.title.y = element_text(face= "plain", size = 12, margin = margin(t=0, r = 10, b = 0, l = 0)),
            legend.position="bottom"
             ) + 
       scale_fill_manual(labels = c("Total", "Escape diapause"), 
                         values=c("grey", "orange"), 
                         name = "Relative population") +
       scale_color_manual(labels = c("Total", "Escape diapause"), 
                          values=c("grey", "orange"), 
                          guide = FALSE) +
       stat_summary(geom="ribbon", fun.y=function(z) { quantile(z,0.5) }, 
                    fun.ymin=function(z) { 0 }, 
                    fun.ymax=function(z) { quantile(z,0.9) }, alpha=0.7)+
       scale_x_continuous(limits = c(0, 5000)) + 
       scale_y_continuous(limits = c(0, 20)) 
  
  # ann_text <- data.frame(CumulativeDDF=700, value=18, 
  #                        CountyGroup=factor("Cooler Areas", 
  #                        levels = c("Cooler Areas","Warmer Areas")))
  #pp + geom_text(data = ann_text, label = "Text")
  plot_name = paste0("diapause_rel_", version, "_", ii, ".png")
  ggsave(plot_name, pp, device="png", path=plot_path, width=10, height=7, unit="in", dpi=400)
}

input_dir = "/Users/hn/Desktop/Kirti/check_point/my_aeolus_2015/diapause_sens/"
plot_path = input_dir

#### Relative
#for (ii in 1:7){
#  file_name_extension = paste0("diapause_rel_rcp85_", ii, ".rds")
#  version = "85"
#  plot_rel_diapause(input_dir, file_name_extension, version, plot_path, ii)
  
#  file_name_extension = paste0("diapause_rel_rcp45_", ii, ".rds")
#  version = "45"
#  plot_rel_diapause(input_dir, file_name_extension, version, plot_path, ii)
  
  #### Absolute
#  file_name_extension = paste0("diapause_abs_rcp85_", ii, ".rds")
#  version = "85"
#  plot_abs_diapause(input_dir, file_name_extension, version, plot_path, ii)
  
#  file_name_extension = paste0("diapause_abs_rcp45_", ii, ".rds")
#  version = "45"
#  plot_abs_diapause(input_dir, file_name_extension, version, plot_path, ii)
  
#}

data_dir = "/Users/hn/Desktop/Kirti/check_point/my_aeolus_2015/diapause_sens/"
plot_path = data_dir
file_list = list.files(path = data_dir, pattern = "diapause_abs", 
                       all.files = FALSE, 
                       full.names = FALSE, 
                       recursive = FALSE)

for (file in file_list){
  version = unlist(strsplit(file, "_"))[3]
  if (version=="rcp45"){ max_y=200} else { max_y=250}
  ii = substr(unlist(strsplit(file, "_"))[4], 1, 1)
  plot_abs_diapause(input_dir, file, version, plot_path, ii, max_y)
}

##########################################
file_list = list.files(path = data_dir, pattern = "diapause_rel", 
                       all.files = FALSE, 
                       full.names = FALSE, 
                       recursive = FALSE)

plot_path = paste0(data_dir, "rel_pop/")
for (file in file_list){
  version = unlist(strsplit(file, "_"))[3]
  ii = substr(unlist(strsplit(file, "_"))[4], 1, 1)
  plot_rel_diapause(input_dir, file, version, plot_path, ii)

}


