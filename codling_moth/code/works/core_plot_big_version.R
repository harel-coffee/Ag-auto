#!/share/apps/R-3.2.2_gcc/bin/Rscript
library(chron)
library(data.table)
library(ggplot2)
library(reshape2)
library(dplyr)
library(foreach)
library(iterators)

source_path = "/home/hnoorazar/cleaner_codes/core.R"
source(source_path)

################################################################################
## *****************               Plotting Codes          *********************
################################################################################

##########################################
#######
####### Bloom Plots
#######
##########################################
plot_bloom_filling <- function(data_dir, file_name = "vertdd_combined_CMPOP_", 
                               version, plot_path, output_name, 
                               x_limits = c(45, 150)){
  output_name = paste0(output_name, "_", version, ".png")
  filename <- paste0(data_dir, file_name, version, ".rds")

  data <- data.table(readRDS(filename))
  data$CountyGroup = as.character(data$CountyGroup)
  data[CountyGroup == 1]$CountyGroup = 'Cooler Areas'
  data[CountyGroup == 2]$CountyGroup = 'Warmer Areas'

  d1 = subset(data, select = c("latitude", "longitude", 
                               "CountyGroup", "ClimateGroup", "ClimateScenario", 
                               "year", "month", "day", "dayofyear", 
                               "vert_Cum_dd_F", 
                               "cripps_pink", "gala", "red_deli"))

  d1 = melt(d1, id = c("latitude", "longitude", 
                       "CountyGroup", "ClimateGroup", "ClimateScenario", 
                       "year", "month", "day", "dayofyear", 
                       "vert_Cum_dd_F"))

  d1[variable == "red_deli"]$variable = "Red Delicious"
  d1[variable == "gala"]$variable = "Gala"
  d1[variable == "cripps_pink"]$variable = "Cripps Pink"

  p1 = ggplot(d1, aes(x=dayofyear, y=value, fill=factor(ClimateGroup))) +
                  stat_summary(geom="ribbon", fun.y=function(z) { quantile(z,0.5) }, fun.ymin=function(z) { quantile(z,0.1) }, fun.ymax=function(z) { quantile(z,0.9) }, alpha=0.3) +
                  stat_summary(geom="ribbon", fun.y=function(z) { quantile(z,0.5) }, fun.ymin=function(z) { quantile(z,0.25) }, fun.ymax=function(z) { quantile(z,0.75) }, alpha=0.7) +
                  stat_summary(geom="line", fun.y=function(z) { quantile(z,0.5) }, size = 1)+
                  scale_color_manual(values=c(rgb(29, 67, 111, max=255), rgb(92, 160, 201, max=255), rgb(211, 91, 76, max=255), rgb(125, 7, 37, max=255)))+
                  scale_fill_manual(values =c(rgb(29, 67, 111, max=255), rgb(92, 160, 201, max=255), rgb(211, 91, 76, max=255), rgb(125, 7, 37, max=255)))+
                  facet_grid(. ~ variable ~ CountyGroup, scales = "free") +
                  scale_x_continuous(breaks=seq(x_limits[1], x_limits[2], 10), limits = x_limits) +
                  theme_bw() +
                  labs(x = "Julian Day", y = "Proportion Full Bloom Completed", fill = "Climate Group") +
                  theme(panel.grid.major = element_line(size=0.2),
                        legend.title = element_text(face="plain", size=12),
                        legend.text = element_text(size=10),
                        legend.position = "bottom",
                        strip.text = element_text(size=12, face="plain"),
                        axis.text = element_text(face="plain", size=10),
                        axis.title.x = element_text(face= "plain", size=16, margin = margin(t=10, r=0, b=0, l=0)),
                        axis.title.y = element_text(face="plain", size=16, margin = margin(t=0, r=10, b=0, l=0))
                    )
  ggsave(output_name, p1, path=plot_path)
}

plot_bloom <- function(data_dir, file_name = "vertdd_combined_CMPOP_", version, 
                       plot_path, output_name, x_limits = c(45, 150)){
  output_name = paste0(output_name, "_", version, ".png")
  filename <- paste0(data_dir, file_name, version, ".rds")
  
  data <- data.table(readRDS(filename))
  data$CountyGroup = as.character(data$CountyGroup)
  data[CountyGroup == 1]$CountyGroup = 'Cooler Areas'
  data[CountyGroup == 2]$CountyGroup = 'Warmer Areas'

  d1 = subset(data, select = c("latitude", "longitude", 
                               "CountyGroup", "ClimateGroup", "ClimateScenario", 
                               "year", "month", "day", "dayofyear", 
                               "vert_Cum_dd_F", 
                               "cripps_pink", "gala", "red_deli"))

  d1 = melt(d1, id = c("latitude", "longitude", 
                       "CountyGroup", "ClimateGroup", "ClimateScenario", 
                       "year", "month", "day", "dayofyear", 
                       "vert_Cum_dd_F"))

  d1[variable == "red_deli"]$variable = "Red Delicious"
  d1[variable == "gala"]$variable = "Gala"
  d1[variable == "cripps_pink"]$variable = "Cripps Pink"

  p1 = ggplot(d1, aes(x=dayofyear, y=value, fill=factor(ClimateGroup))) +
    #stat_summary(geom="ribbon", fun.y=function(z) { quantile(z,0.5) }, fun.ymin=function(z) { quantile(z,0.1) }, fun.ymax=function(z) { quantile(z,0.9) }, alpha=0.3) +
    #stat_summary(geom="ribbon", fun.y=function(z) { quantile(z,0.5) }, fun.ymin=function(z) { quantile(z,0.25) }, fun.ymax=function(z) { quantile(z,0.75) }, alpha=0.7) +
    stat_summary(geom="line", fun.y=function(z) { quantile(z,0.5) }, aes(color=factor(ClimateGroup)), size = 1)+ #, aes(color=factor(Timeframe))) + , # aes(color=factor(ClimateGroup))
    scale_color_manual(values=c(rgb(29, 67, 111, max=255), rgb(92, 160, 201, max=255), rgb(211, 91, 76, max=255), rgb(125, 7, 37, max=255)))+#c("black", "red","darkgreen","blue")) +
    #scale_fill_manual(values=c(rgb(29, 67, 111, max=255), rgb(92, 160, 201, max=255), rgb(211, 91, 76, max=255), rgb(125, 7, 37, max=255)))+#c("black", "red","darkgreen","blue")) +
    facet_grid(. ~ variable ~ CountyGroup, scales = "free") +
    #xlim(45, 165) +
    scale_x_continuous(breaks=seq(x_limits[1], x_limits[2], 10), limits = x_limits) +
    theme_bw() +
    labs(x = "Julian Day", y = "Proportion Completing Full Bloom", color = "Climate Group") +
    theme(panel.grid.major = element_line(size=0.2),
          # axis.title = element_text(face = "plain", size = 16, margin=margin(2)),
          legend.title = element_text(face="plain", size=12),
          legend.text = element_text(size=10),
          legend.position = "bottom",
          #plot.title = element_text(face = "bold", size = 18, hjust = 0.5),
          strip.text = element_text(size=12, face="plain"),
          axis.text = element_text(face="plain", size=10),
          axis.title.x = element_text(face="plain", size=16, margin=margin(t=10, r=0, b=0, l=0)),
          axis.title.y = element_text(face="plain", size=16, margin=margin(t=0, r=10, b=0, l=0))
    )
  ggsave(output_name, p1, path=plot_path, width=7, height=7, unit="in")
}

###############################################################
#######
###############################################################

plot_cumdd_eggHatch <- function(input_dir, file_name ="combined_CMPOP_", 
                                version, output_dir, output_type="cumdd"){
  out_name = paste0("plot_", output_type, "_", version ,".png")
  #############################################
  ###    Egg Hatch
  #############################################
  if (output_type == "eggHatch"){
    data = compute_cumdd_eggHatch(input_dir=data_dir, file_name="combined_CMPOP_", version)
    data$CountyGroup = as.character(data$CountyGroup)
    data[CountyGroup == 1]$CountyGroup = 'Cooler Areas'
    data[CountyGroup == 2]$CountyGroup = 'Warmer Areas'
    data = melt(data, id = c("ClimateGroup", "CountyGroup", 
                             "latitude", "longitude", 
                             "ClimateScenario", "year", "dayofyear"))

    plot = ggplot(data[value >=0.01 & value <.98 & dayofyear <300], 
                  aes(x=dayofyear, y=value, fill=factor(variable))) +
    #geom_line(aes(fill=factor(Timeframe), color=factor(Timeframe) )) +
    stat_summary(geom="ribbon", 
                 fun.y=function(z) { quantile(z,0.5) }, 
                 fun.ymin=function(z) { quantile(z,0.1) }, 
                 fun.ymax=function(z) { quantile(z,0.9) }, 
                 alpha=0.3) +
    stat_summary(geom="ribbon", 
                 fun.y=function(z) { quantile(z,0.5) }, 
                 fun.ymin=function(z) { quantile(z,0.25) }, 
                 fun.ymax=function(z) { quantile(z,0.75) }, 
                 alpha=0.8) + 
    stat_summary(geom="line", 
                 fun.y=function(z) { quantile(z,0.5) })+
    
    scale_color_manual(values=c(rgb(29, 67, 111, max=255), 
                                rgb(92, 160, 201, max=255), 
                                rgb(211, 91, 76, max=255), 
                                rgb(125, 7, 37, max=255)),
                      labels = c("Gen. 1", "Gen. 2", "Gen. 3", "Gen. 4")) +
    scale_fill_manual(values=c(rgb(29, 67, 111, max=255), 
                               rgb(92, 160, 201, max=255), 
                               rgb(211, 91, 76, max=255), 
                               rgb(125, 7, 37, max=255)),
                      labels = c("Gen. 1", "Gen. 2", "Gen. 3", "Gen. 4"))+
    
    facet_grid(. ~ ClimateGroup ~ CountyGroup, scales="free") +
    scale_x_continuous(breaks=seq(0, 300, 50)) +
    theme_bw() +
    geom_vline(xintercept=c(100, 150, 200, 250, 300), linetype="solid", color ="grey", size=0.2) +
    geom_hline(yintercept=c(.25, .5, .75), linetype="solid", color ="grey", size=0.2) +
    geom_vline(xintercept=c(120, 226), linetype="solid", color ="red") +
    labs(x = "Julian Day", y = "Cumulative Population Fraction", fill = "Larva Generation") +
    theme(
      #panel.grid.major = element_line(size = 0.2),
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      legend.title = element_text(face="plain", size=12),
      legend.text = element_text(size=10),
      legend.position = "bottom",
      strip.text = element_text(size=12, face="plain"),
      axis.text = element_text(face="plain", size=10),
      axis.title.x = element_text(face="plain", size=16, margin=margin(t=10, r=0, b=0, l=0)),
      axis.title.y = element_text(face="plain", size=16, margin=margin(t=0, r=10, b=0, l=0)))
  
  ggsave(out_name, plot, path=output_dir, width=7, height=7, unit="in")
  }

  #############################################
  ### cumdd
  #############################################
  if (output_type == "cumdd"){
    filename = paste0(input_dir, file_name, version, ".rds")
    data <- data.table(readRDS(filename))

    data$CountyGroup = as.character(data$CountyGroup)
    data[CountyGroup == 1]$CountyGroup = 'Cooler Areas'
    data[CountyGroup == 2]$CountyGroup = 'Warmer Areas'
    data = subset(data, select = c("ClimateGroup", "CountyGroup", 
                                   "latitude", "longitude", 
                                   "ClimateScenario", 
                                   "year", "dayofyear", "CumDDinF"))
    data$CumDD = data$CumDDinF

    data <- data[, .(CumDD = median(CumDDinF)), by = c("CountyGroup", "latitude", "longitude", 
                                                       "ClimateScenario", "ClimateGroup", "dayofyear")]
    if (version == "rcp85"){
      y_range = seq(0, 5750, 500)
    }
    else{
      y_range=seq(0, 4500, 500)
    }
    plot = ggplot(data, aes(x=dayofyear, y=CumDD, fill=factor(ClimateGroup))) +
           #geom_line(aes(fill=factor(Timeframe), color=factor(Timeframe) )) +
           stat_summary(geom="ribbon", fun.y=function(z) { quantile(z,0.5) }, 
                                       fun.ymin=function(z) { quantile(z,0.1) }, 
                                       fun.ymax=function(z) { quantile(z,0.9) }, alpha=0.4) +
           stat_summary(geom="ribbon", fun.y=function(z) { quantile(z,0.5) }, 
                                       fun.ymin=function(z) { quantile(z,0.25) }, 
                                       fun.ymax=function(z) { quantile(z,0.75) }, alpha=0.8) + 
           stat_summary(geom="line", fun.y=function(z) { quantile(z,0.5) })+ #, aes(color=factor(Timeframe))) + 
        
           scale_color_manual(values=c(rgb(29, 67, 111, max=255), 
                                       rgb(92, 160, 201, max=255), 
                                       rgb(211, 91, 76, max=255), 
                                       rgb(125, 7, 37, max=255)))+#c("black", "red","darkgreen","blue")) +
           scale_fill_manual(values=c(rgb(29, 67, 111, max=255), 
                                      rgb(92, 160, 201, max=255), 
                                      rgb(211, 91, 76, max=255), 
                                      rgb(125, 7, 37, max=255)))+#c("black", "red","darkgreen","blue")) +
        
           facet_grid(. ~ CountyGroup, scales="free") +
           scale_x_continuous(breaks=seq(0, 370, 50)) +
           scale_y_continuous(breaks=y_range) +
           theme_bw() +
           labs(x = "Julian Day", y = "Cumulative Degree Days (in F)", fill = "Climate Group") +
           theme(
                # panel.grid.major = element_line(size = 0.7),
                # panel.grid.major = element_blank(),
                panel.grid.minor = element_blank(),
                legend.title = element_text(face = "plain", size = 12),
                legend.text = element_text(size = 10),
                legend.position = "bottom",
                strip.text = element_text(size = 12, face = "plain"),
                axis.text = element_text(face = "plain", size = 10),
                axis.title.x = element_text(face = "plain", size=16, margin=margin(t=10, r=0, b=0, l=0)),
                axis.title.y = element_text(face = "plain", size=16, margin=margin(t=0, r=10, b=0, l=0)))

    out_name = paste0("plot_", output_type, "_", version, ".png")
    ggsave(out_name, plot, path=output_dir)
  }

  if (output_type==3){
    filename = paste0(input_dir, file_name, version, ".rds")
    data <- data.table(readRDS(filename))

    data$CountyGroup = as.character(data$CountyGroup)
    data[CountyGroup == 1]$CountyGroup = 'Cooler Areas'
    data[CountyGroup == 2]$CountyGroup = 'Warmer Areas'
    data = subset(data, select = c("ClimateGroup", "CountyGroup", "latitude", 
                                   "longitude", "ClimateScenario", "year", "dayofyear", "CumDDinF"))
    data$CumDD = data$CumDDinF

    plot = ggplot(data, aes(x=dayofyear, y=CumDD, fill=factor(ClimateGroup))) +
    #geom_line(aes(fill=factor(Timeframe), color=factor(Timeframe) )) +
    stat_summary(geom="ribbon", fun.y=function(z) { quantile(z,0.5) }, 
                                fun.ymin=function(z) { quantile(z,0.1) }, 
                                fun.ymax=function(z) { quantile(z,0.9) }, alpha=0.4) +
    stat_summary(geom="ribbon", fun.y=function(z) { quantile(z,0.5) }, 
                                fun.ymin=function(z) { quantile(z,0.25) }, 
                                fun.ymax=function(z) { quantile(z,0.75) }, alpha=0.8) + 
    stat_summary(geom="line", fun.y=function(z) { quantile(z,0.5) })+ #, aes(color=factor(Timeframe))) + 
    
    scale_color_manual(values=c(rgb(29, 67, 111, max=255), rgb(92, 160, 201, max=255), 
                                rgb(211, 91, 76, max=255), rgb(125, 7, 37, max=255)))+#c("black", "red","darkgreen","blue")) +
    scale_fill_manual(values=c(rgb(29, 67, 111, max=255), rgb(92, 160, 201, max=255), 
                               rgb(211, 91, 76, max=255), rgb(125, 7, 37, max=255)))+#c("black", "red","darkgreen","blue")) +
    
    facet_grid(CountyGroup ~ ClimateGroup ~ ., scales = "fixed") +
    scale_x_continuous(breaks=seq(0, 370, 50)) +
    scale_y_continuous(breaks=seq(0, 5000, 1000)) +
    theme_bw() +
    labs(x = "Julian Day", y = "Cumulative Degree Days (in F)", fill = "Climate Group") +
    theme(
      panel.grid.major = element_line(size = 0.7),
      
      legend.title = element_text(face = "plain", size = 16),
      legend.text = element_text(size = 12),
      legend.position = "bottom",
      
      strip.text = element_text(size= 12, face = "plain"),
      axis.text = element_text(face="plain", size = 10),
      axis.title.x = element_text(face= "plain", size = 16, margin = margin(t = 10, r = 0, b = 0, l = 0)),
      axis.title.y = element_text(face= "plain", size = 16, margin = margin(t = 0, r = 10, b = 0, l = 0)))
  out_name = paste0("plot_cumdd_" , version, "_type", output_type ,".png")
  ggsave(out_name, plot, path=output_dir, width=7, height=7, unit="in")
  }
}

#####################################################################################
#######################                                   ###########################
#######################          Diapause Plots           ###########################
#######################                                   ###########################
#####################################################################################
plot_diapause <- function(input_dir, file_name_extension, version, pop_type, plot_path){
  # input_dir
  # file_name_extension: file name including extention .rds
  # version
  # pop_type either abs or rel
  # plot_path
  if (pop_type == "abs"){plot_abs_diapause(input_dir, file_name_extension, version, plot_path)}
  else {plot_rel_diapause(input_dir, file_name_extension, version, plot_path)}
}

plot_abs_diapause <- function(input_dir, file_name_extension, version, plot_path){
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
  
  diap_plot <- ggplot(data, aes(x=CumulativeDDF, y=value, color=variable, fill=factor(variable))) + 
               geom_vline(xintercept=c(213, 1153, 2313, 3443, 4453), linetype="solid", color ="grey", size=.25) +
               geom_hline(yintercept=c(25, 50, 75, 100), linetype="solid", color ="grey", size=.25) +
               annotate(geom="text", x=700,  y=85, label="Gen 1", color="black", angle=30) +
               annotate(geom="text", x=1700, y=80, label="Gen 2", color="black", angle=30) + 
               annotate(geom="text", x=2900, y=75, label="Gen 3", color="black", angle=30) + 
               annotate(geom="text", x=3920, y=70, label="Gen 4", color="black", angle=30) +
               theme_bw() +
               facet_grid(. ~ CountyGroup ~ ClimateGroup, scales = "free") +
               labs(x = "Cumulative Degree Days (in F)", y = "Absolute Population", color = "Absolute Population") +
               theme(axis.text = element_text(face= "plain", size = 8),
                     panel.grid.major = element_blank(),
                     panel.grid.minor = element_blank(),
                     axis.title.x = element_text(face= "plain", size = 12, margin = margin(t=10, r = 0, b = 0, l = 0)),
                     axis.title.y = element_text(face= "plain", size = 12, margin = margin(t=0, r = 10, b = 0, l = 0)),
                     legend.position="bottom"
               ) + 
               scale_fill_manual(labels = c("Total", "Escape Diapause"), values=c("grey", "orange"), name = "Absolute Population") +
               scale_color_manual(labels = c("Total", "Escape Diapause"), values=c("grey", "orange"), guide = FALSE) +
               stat_summary(geom="ribbon", fun.y=function(z) { quantile(z,0.5) }, 
                             fun.ymin=function(z) { 0 }, 
                             fun.ymax=function(z) { quantile(z,0.9) }, alpha=0.7)+
               scale_x_continuous(limits = c(0, 5000)) + 
               scale_y_continuous(limits = c(0, 100))
  
  plot_name = paste0("diapause_abs_", version, ".png")
  ggsave(plot_name, diap_plot, device="png", path=plot_path, width=10, height=7, unit="in")
}

plot_rel_diapause <- function(input_dir, file_name_extension, version, plot_path){
  file_name = paste0(input_dir, file_name_extension)
  data <- data.table(readRDS(file_name))
  data$CountyGroup = as.character(data$CountyGroup)
  data[CountyGroup == 1]$CountyGroup = 'Cooler Areas'
  data[CountyGroup == 2]$CountyGroup = 'Warmer Areas'
  data <- data[variable =="RelLarvaPop" | variable =="RelNonDiap"]
  data <- subset(data, select=c("ClimateGroup", "CountyGroup", "CumulativeDDF", "variable", "value"))
  data$variable <- factor(data$variable)

  pp = ggplot(data, aes(x=CumulativeDDF, y=value, color=variable, fill=factor(variable))) + 
       geom_vline(xintercept=c(213, 1153, 2313, 3443, 4453), linetype="solid", color ="grey", size=.25) +
       geom_hline(yintercept=c(5, 10, 15, 20), linetype="solid", color ="grey", size=.25) +
       annotate(geom="text", x=700,  y=18, label="Gen 1", color="black", angle=30) +
       annotate(geom="text", x=1700, y=16, label="Gen 2", color="black", angle=30) + 
       annotate(geom="text", x=2900, y=14, label="Gen 3", color="black", angle=30) + 
       annotate(geom="text", x=3920, y=12, label="Gen 4", color="black", angle=30) +
       theme_bw() +
       facet_grid(. ~ CountyGroup ~ ClimateGroup, scales = "free") +
       labs(x = "Cumulative Degree Days(in F)", y = "Relative Population", color = "Relative Population") +
       theme(axis.text = element_text(face= "plain", size = 8),
             panel.grid.major = element_blank(),
             panel.grid.minor = element_blank(),
             axis.title.x = element_text(face= "plain", size = 12, margin = margin(t=10, r = 0, b = 0, l = 0)),
             axis.title.y = element_text(face= "plain", size = 12, margin = margin(t=0, r = 10, b = 0, l = 0)),
             legend.position="bottom"
       ) + 
       scale_fill_manual(labels = c("Total", "Escape Diapause"), values=c("grey", "orange"), name = "Relative Population") +
       scale_color_manual(labels = c("Total", "Escape Diapause"), values=c("grey", "orange"), guide = FALSE) +
       stat_summary(geom="ribbon", fun.y=function(z) { quantile(z,0.5) }, 
                                   fun.ymin=function(z) { 0 }, 
                                   fun.ymax=function(z) { quantile(z,0.9) }, alpha=0.7)+
       scale_x_continuous(limits = c(0, 5000)) + 
       scale_y_continuous(limits = c(0, 20)) 
  
  #ann_text <- data.frame(CumulativeDDF=700, value=18, CountyGroup=factor("Cooler Areas", levels = c("Cooler Areas","Warmer Areas")))
  #pp + geom_text(data = ann_text, label = "Text")
  plot_name = paste0("diapause_rel_", version,".png")
  ggsave(plot_name, pp, device="png", path=plot_path, width=10, height=7, unit="in")
}
############################################################################################################
plot_adult_DoY_filling_median <- function(input_dir, file_name ="combined_CMPOP_", 
                                   version, output_dir){
  out_name = paste0("plot_Adult_DoY_median_", version ,".png")
  #############################################
  ###    Adult Emergence
  #############################################
  data = compute_cumdd_adult_emergence_median(input_dir=data_dir, 
                                              file_name="combined_CMPOP_", 
                                              version = version)
  data$CountyGroup = as.character(data$CountyGroup)
  data[CountyGroup == 1]$CountyGroup = 'Cooler Areas'
  data[CountyGroup == 2]$CountyGroup = 'Warmer Areas'
  data = melt(data, id = c("ClimateGroup", "CountyGroup", 
                           "latitude", "longitude", 
                           "ClimateScenario", "year", "dayofyear"))

  plot = ggplot(data[value >=0.01 & value <.98 & dayofyear <360], 
                aes(x=dayofyear, y=value, fill=factor(variable))) +
  #geom_line(aes(fill=factor(Timeframe), color=factor(Timeframe) )) +
  stat_summary(geom="ribbon", 
               fun.y=function(z) { quantile(z,0.5) }, 
               fun.ymin=function(z) { quantile(z,0.1) }, 
               fun.ymax=function(z) { quantile(z,0.9) }, 
               alpha=0.3) +
  stat_summary(geom="ribbon", 
               fun.y=function(z) { quantile(z,0.5) }, 
               fun.ymin=function(z) { quantile(z,0.25) }, 
               fun.ymax=function(z) { quantile(z,0.75) }, 
               alpha=0.8) + 
  stat_summary(geom="line", 
               fun.y=function(z) { quantile(z,0.5) })+
  
  scale_color_manual(values=c(rgb(29, 67, 111, max=255), 
                              rgb(92, 160, 201, max=255), 
                              rgb(211, 91, 76, max=255), 
                              rgb(125, 7, 37, max=255)),
                    labels = c("Gen. 1", "Gen. 2", "Gen. 3", "Gen. 4")) +
  scale_fill_manual(values=c(rgb(29, 67, 111, max=255), 
                             rgb(92, 160, 201, max=255), 
                             rgb(211, 91, 76, max=255), 
                             rgb(125, 7, 37, max=255)),
                    labels = c("Gen. 1", "Gen. 2", "Gen. 3", "Gen. 4"))+
  
  facet_grid(. ~ ClimateGroup ~ CountyGroup, scales="free") +
  scale_x_continuous(breaks=seq(0, 300, 50)) +
  theme_bw() +
  geom_vline(xintercept=c(100, 150, 200, 250, 300), linetype="solid", color ="grey", size=0.2) +
  geom_hline(yintercept=c(.25, .5, .75), linetype="solid", color ="grey", size=0.2) +
  # geom_vline(xintercept=c(120, 226), linetype="solid", color ="red") +
  labs(x = "Day of Year", y = paste0("Cumulative Adult Emergence"), fill = "Adult Generation") +
  theme(
    #panel.grid.major = element_line(size = 0.2),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    legend.title = element_text(face="plain", size=12),
    legend.text = element_text(size=10),
    legend.position = "bottom",
    strip.text = element_text(size=12, face="plain"),
    axis.text = element_text(face="plain", size=10),
    axis.title.x = element_text(face="plain", size=16, margin=margin(t=10, r=0, b=0, l=0)),
    axis.title.y = element_text(face="plain", size=16, margin=margin(t=0, r=10, b=0, l=0)))

  ggsave(out_name, plot, path=output_dir)
  }

plot_adult_DoY_filling_mean <- function(input_dir, file_name ="combined_CMPOP_", 
                                   version, output_dir){
  out_name = paste0("plot_Adult_DoY_mean_", version ,".png")
  #############################################
  ###    Adult Emergence
  #############################################
  data = compute_cumdd_adult_emergence_mean(input_dir=data_dir, 
                                              file_name="combined_CMPOP_", 
                                               version = version)
  data$CountyGroup = as.character(data$CountyGroup)
  data[CountyGroup == 1]$CountyGroup = 'Cooler Areas'
  data[CountyGroup == 2]$CountyGroup = 'Warmer Areas'
  data = melt(data, id = c("ClimateGroup", "CountyGroup", 
                           "latitude", "longitude", 
                           "ClimateScenario", "year", "dayofyear"))

  plot = ggplot(data[value >=0.01 & value <.98 & dayofyear <360], 
                aes(x=dayofyear, y=value, fill=factor(variable))) +
  #geom_line(aes(fill=factor(Timeframe), color=factor(Timeframe) )) +
  stat_summary(geom="ribbon", 
               fun.y=function(z) { quantile(z,0.5) }, 
               fun.ymin=function(z) { quantile(z,0.1) }, 
               fun.ymax=function(z) { quantile(z,0.9) }, 
               alpha=0.3) +
  stat_summary(geom="ribbon", 
               fun.y=function(z) { quantile(z,0.5) }, 
               fun.ymin=function(z) { quantile(z,0.25) }, 
               fun.ymax=function(z) { quantile(z,0.75) }, 
               alpha=0.8) + 
  stat_summary(geom="line", 
               fun.y=function(z) { quantile(z,0.5) })+
  
  scale_color_manual(values=c(rgb(29, 67, 111, max=255), 
                              rgb(92, 160, 201, max=255), 
                              rgb(211, 91, 76, max=255), 
                              rgb(125, 7, 37, max=255)),
                    labels = c("Gen. 1", "Gen. 2", "Gen. 3", "Gen. 4")) +
  scale_fill_manual(values=c(rgb(29, 67, 111, max=255), 
                             rgb(92, 160, 201, max=255), 
                             rgb(211, 91, 76, max=255), 
                             rgb(125, 7, 37, max=255)),
                    labels = c("Gen. 1", "Gen. 2", "Gen. 3", "Gen. 4"))+
  
  facet_grid(. ~ ClimateGroup ~ CountyGroup, scales="free") +
  scale_x_continuous(breaks=seq(0, 350, 50)) +
  theme_bw() +
  geom_vline(xintercept=c(100, 150, 200, 250, 300, 350), linetype="solid", color ="grey", size=0.2) +
  geom_hline(yintercept=c(.25, .5, .75), linetype="solid", color ="grey", size=0.2) +
  # geom_vline(xintercept=c(120, 226), linetype="solid", color ="red") +
  labs(x = "Day of Year", y = paste0("Cumulative Adult Emergence"), fill = "Adult Generation") +
  theme(
    #panel.grid.major = element_line(size = 0.2),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    legend.title = element_text(face="plain", size=12),
    legend.text = element_text(size=10),
    legend.position = "bottom",
    strip.text = element_text(size=12, face="plain"),
    axis.text = element_text(face="plain", size=10),
    axis.title.x = element_text(face="plain", size=16, margin=margin(t=10, r=0, b=0, l=0)),
    axis.title.y = element_text(face="plain", size=16, margin=margin(t=0, r=10, b=0, l=0)))

  ggsave(out_name, plot, path=output_dir)
  }
####################################################################################################
##################                                                                ##################
##################                       Box Plots                                ##################
##################                                                                ##################
####################################################################################################

##################
##################   Adult Emergence
##################
plot_adult_emergence <- function(input_dir, file_name, 
                                 box_width=.25, plot_path, output_name, 
                                 color_ord = c("grey70", "dodgerblue", "olivedrab4", "red")
){
  #
  # These plots are produced by combined_CM files.
  #
  output_name = paste0(output_name, ".png")
  file_name <- paste0(input_dir, file_name)
  data <- data.table(readRDS(file_name))
  data <- subset(data, select = c("Emergence", "ClimateGroup", 
                                  "ClimateScenario", 
                                  "CountyGroup"))
  
  data$CountyGroup = as.character(data$CountyGroup)
  data[CountyGroup == 1]$CountyGroup = 'Cooler Areas'
  data[CountyGroup == 2]$CountyGroup = 'Warmer Areas'
  
  data = data[, .(Emergence = Emergence),
              by = c("ClimateGroup", "CountyGroup")]
  
  data <- subset(data, select = c("ClimateGroup", "CountyGroup", "Emergence"))
  ######
  ###### Compute medians of each group to annotate in the plot, if possible!!!
  ######
  df <- data.frame(data)
  df <- (df %>% group_by(CountyGroup, ClimateGroup))
  medians <- (df %>% summarise(med = median(Emergence)))
  medians_vec <- medians$med
  
  p = ggplot(data = data, aes(x=ClimateGroup, y=Emergence, fill=ClimateGroup))+
      geom_boxplot(outlier.size=-.15, notch=TRUE, width=box_width, lwd=.25) +
      theme_bw() +
      scale_x_discrete(expand=c(0, 2), limits = levels(data$ClimateGroup[1])) +
      scale_y_continuous(breaks = round(seq(40, 170, by = 10), 1)) +
      labs(x="Time Period", y="Day of Year", color = "Climate Group") +
      facet_wrap(~CountyGroup) +
      theme(legend.position="bottom", 
            legend.key.size = unit(.75,"line"),
            panel.grid.minor = element_blank(),
            panel.spacing=unit(.5,"cm"),
            legend.text=element_text(size=5),
            legend.margin=margin(t = -.1, r = 0, b = 0, l = 0, unit = 'cm'),
            # plot.margin = unit(c(0.5, 0.5, 0.5, 0.5), "cm"),
            legend.title = element_blank(),
            panel.grid.major = element_line(size = 0.1),
            axis.text = element_text(face = "plain", size = 10),
            axis.title.x = element_text(face = "plain", size=8, margin = margin(t=5, r=0, b=0, l=0)),
            axis.text.x = element_text(size = 5),
            axis.title.y = element_text(face = "plain", size=8, margin = margin(t=0, r=1, b=0, l=0)),
            axis.text.y  = element_blank(),
            axis.ticks.y = element_blank()
      ) +
      scale_fill_manual(values=color_ord,
                        name="Time\nPeriod", 
                        labels=c("Historical","2040","2060","2080")) + 
      scale_color_manual(values=color_ord,
                         name="Time\nPeriod", 
                         limits = color_ord,
                         labels=c("Historical","2040","2060","2080")) + 
      geom_text(data = medians, 
                aes(label = sprintf("%1.0f", medians$med), y=medians$med), 
                size=1.5, 
                position =  position_dodge(.09),
                vjust = -1.5) +
      #stat_summary(geom="text", fun.y=quantile,
      #             aes(label=sprintf("%1.1f", ..x..), color=factor(ClimateGroup)),
      #             position=position_nudge(x=0.33), size=.5) +
      coord_flip()
  ggsave(output_name, p, path=plot_path, width=5.5, height=3.1, unit="in")
}
##################
##################   Generations of Adults or Larva by Aug 23
##################
plot_No_generations <- function(input_dir,
                                file_name,
                                stage,
                                dead_line,
                                box_width=.25,
                                plot_with = 6.5,
                                plot_height = 2.5,
                                plot_path,
                                version,
                                color_ord = c("grey70", "dodgerblue", "olivedrab4", "red")
                                ){
  # stage: either larva or adult
  # version either rcp45 or rcp85
  # 
  # This function does not run on Aeolus with R.3.2.2. 
  # I will produce it on my computer.
  #
  file_name <- paste0(input_dir, file_name)
  data <- data.table(readRDS(file_name))
  data$CountyGroup = as.character(data$CountyGroup)
  data[CountyGroup == 1]$CountyGroup = 'Cooler Areas'
  data[CountyGroup == 2]$CountyGroup = 'Warmer Areas'
  
  if (stage=="Larva"){
    var = "NumLarvaGens"
  } else {
    var = "NumAdultGens"
  }
  
  data <- subset(data, select = c("ClimateGroup", "CountyGroup", var))
  ######
  ###### Compute medians of each group to annotate in the plot, if possible!!!
  ######
  df <- data.frame(data)
  df <- (df %>% group_by(CountyGroup, ClimateGroup))
  medians <- (df %>% summarise(med = median(!!sym(var))))
  rm(df)
  if (dead_line=="Aug"){
    y_lab = paste0("Number of ", stage, " Generations by August 23")
  }else{
    y_lab = paste0("Number of ", stage, " Generations by November 5")
  }
  box_plot = ggplot(data = data, aes(x = ClimateGroup, y = !!sym(var), fill = ClimateGroup)) + 
             geom_boxplot(outlier.size=-.15, lwd=0.25, notch=TRUE, width=box_width) +
             # The bigger the number in expand below, the smaller the space between y-ticks
             scale_x_discrete(expand=c(0, 3), limits = levels(data$ClimateGroup[1])) +
             scale_y_continuous(limits = c(.5, 4), breaks=seq(1, 5, by=1)) + 
             theme_bw() +
             labs(x="Time Period", 
                  y=y_lab, 
                  color = "Climate Group") +
             facet_wrap(~CountyGroup) +
             theme(legend.position="bottom", 
                	 legend.key.size = unit(.75,"line"),
                   legend.text=element_text(size=5),
                   legend.margin=margin(t=-.1, r=0, b=0, l=0, unit='cm'),
                   # plot.margin = unit(c(0.5, 0.5, 0.5, 0.5), "cm"),
                   legend.title = element_blank(),
                   #panel.grid.major = element_line(size = 0.1),
                   #panel.grid.major = element_blank(),
                   panel.grid.minor = element_blank(),
                   axis.text = element_text(face = "plain", size = 10),
                   axis.text.x = element_text(size = 7),
                   axis.title.x = element_text(face = "plain", 
                                               size=8, 
                                               margin = margin(t=3, r=0, b=0, l=0)),
                  
                   axis.title.y = element_text(face = "plain", 
                                               size=8, 
                                               margin=margin(t=0, r=1.5, b=0, l=0)),
                   axis.text.y  = element_blank(),
                   axis.ticks.y = element_blank()
                   ) +
             scale_fill_manual(values=color_ord, name="Time\nPeriod") + 
             scale_color_manual(values=color_ord, name="Time\nPeriod", limits = color_ord) + 
             geom_text(data = medians, 
                       aes(label = sprintf("%1.1f", medians$med), y=medians$med), 
                       size=1.75, 
                       position =  position_dodge(.09),
                       vjust = -1
                       ) +
             coord_flip()
  
  plot_name = paste0(stage, "_Gen_", dead_line, "_", version)
  ggsave(paste0(plot_name, ".png"), 
         box_plot, 
         path=plot_path, 
         device="png", 
         width=plot_with, height=plot_height, units = "in")
}
##################
################## Flight vs. DoY
##################
plot_flight_DoY_half <- function(input_dir, input_name, stage, 
                                 output_dir, output_name, 
                                 plot_with=7, plot_height=3){
  color_ord = c("grey70", "dodgerblue", "olivedrab4", "red")
  data <- readRDS(paste0(input_dir, input_name))
  if (stage == "adult"){
    data <- subset(data, select = c("AGen1_0.5", "AGen2_0.5", "AGen3_0.5", "AGen4_0.5",
                                    "ClimateGroup", "CountyGroup"))
    L = c('AGen1_0.5','AGen2_0.5', 'AGen3_0.5','AGen4_0.5')
  }
  else{
    data <- subset(data, select = c("LGen1_0.5", "LGen2_0.5", "LGen3_0.5", "LGen4_0.5",
                                    "ClimateGroup", "CountyGroup"))
    
    L = c('LGen1_0.5', 'LGen2_0.5',  'LGen3_0.5', 'LGen4_0.5')
  }
  
  data$CountyGroup = as.character(data$CountyGroup)
  data[CountyGroup == 1]$CountyGroup = 'Cooler Areas'
  data[CountyGroup == 2]$CountyGroup = 'Warmer Areas'
  
  data_melted = melt(data, id = c("ClimateGroup", "CountyGroup"))
  data_melted$variable <- factor(data_melted$variable, levels = L, ordered = TRUE)
  
  bplot <- ggplot(data = data_melted, aes(x=variable, y=value), group = variable) + 
    geom_boxplot(outlier.size=-.15, notch=FALSE, width=.4, lwd=.25, aes(fill=ClimateGroup), 
                 position=position_dodge(width=0.5)) + 
    scale_y_continuous(limits = c(80, 370), breaks = seq(100, 360, by = 50)) +
    #geom_vline(xintercept=4.5, linetype="solid", color = "grey", size=1)+
    #geom_vline(xintercept=8.5, linetype="solid", color = "grey", size=1)+
    # annotate("text", x=2.5, y=369, angle=270, label= "boat", size=8, fontface="plain") + 
    
    facet_wrap(~CountyGroup, scales="free", ncol=6, dir="v") + 
    labs(x="Time Period", y="Day of Year", color = "Climate Group", title=factor(data_melted$CountyGroup)) + 
    theme_bw() +
    theme(legend.position="bottom", 
          legend.margin=margin(t=-.1, r=0, b=5, l=0, unit = 'cm'),
          legend.title = element_blank(),
          legend.text = element_text(size=7, face="plain"),
          legend.key.size = unit(.5, "cm"), 
          panel.grid.major = element_line(size = 0.1),
          panel.grid.minor = element_line(size = 0.1),
          strip.text = element_text(size= 6, face = "plain"),
          axis.text = element_text(face = "plain", size = 4),
          axis.title.x = element_text(face = "plain", size = 10, 
                                      margin = margin(t=10, r=0, b=0, l=0)),
          axis.text.x = element_text(size = 6),
          axis.title.y = element_text(face = "plain", size = 10, 
                                      margin = margin(t=0, r=7, b=0, l=0)),
          axis.text.y  = element_blank(),
          axis.ticks.y = element_blank(),
          plot.margin = unit(c(t=-0.35, r=.7, b=-4.7, l=0.3), "cm")
    ) +
    scale_color_manual(values=color_ord,
                       name="Time\nPeriod", 
                       limits = color_ord,
                       labels=c("Historical","2040","2060","2080")) +
    scale_fill_manual(values=color_ord,
                      name="Time\nPeriod", 
                      labels=c("Historical","2040","2060","2080")) + 
    coord_flip()
  #bplot <- add_sub(bplot, label="Gen. 1", x=1.02, y=8, angle=270, size=6, fontface="plain")
  ggsave(output_name, bplot, device="png", path=plot_path, width=plot_with, height=plot_height, unit="in")
}



