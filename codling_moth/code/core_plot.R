.libPaths("/data/hydro/R_libs35")
.libPaths()

library(data.table)
library(dplyr)
library(ggpubr)

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
  print ("line 25, plot_bloom_filling")
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
  print ("line 41, plot_bloom_filling")
  d1[variable == "red_deli"]$variable = "Red Delicious"
  d1[variable == "gala"]$variable = "Gala"
  d1[variable == "cripps_pink"]$variable = "Cripps Pink"
  print ("line 45, plot_bloom_filling")
  color_ord = c("grey70", "dodgerblue", "olivedrab4", "red")
  p1 = ggplot(d1, aes(x=dayofyear, y=value, fill=factor(ClimateGroup))) +
       labs(x = "day of year", y = "proportion full bloom completed", fill = "Climate Group") +
       guides(fill=guide_legend(title="Time period")) + 
       facet_grid(. ~ variable ~ CountyGroup, scales = "free") +
       stat_summary(geom="ribbon", fun.y=function(z) { quantile(z,0.5) }, 
                                   fun.ymin=function(z) { quantile(z,0.1) }, 
                                   fun.ymax=function(z) { quantile(z,0.9) }, alpha=0.3) +
       stat_summary(geom="ribbon", fun.y=function(z) { quantile(z,0.5) }, 
                                   fun.ymin=function(z) { quantile(z,0.25) }, 
                                   fun.ymax=function(z) { quantile(z,0.75) }, alpha=0.7) +
       stat_summary(geom="line", fun.y=function(z) { quantile(z,0.5) }, size = 1)+
       scale_color_manual(values=color_ord) +
       scale_fill_manual(values =color_ord) +
       scale_x_continuous(breaks=seq(x_limits[1], x_limits[2], 10), limits = x_limits) +
       theme(panel.grid.major = element_line(size=0.2),
             panel.spacing=unit(.5, "cm"),
             legend.title = element_blank(),
             strip.text = element_text(face="bold", size=16, color="black"),
             legend.text = element_text(size=10),
             legend.position = "bottom",
             axis.text = element_text(face="plain", size=10, color="black"),
             axis.ticks = element_line(color = "black", size = .2),
             axis.title.x = element_text(face= "bold", size=16, margin = margin(t=10, r=0, b=0, l=0)),
             axis.title.y = element_text(face="bold", size=16, margin = margin(t=0, r=10, b=0, l=0))
             )
  ggsave(output_name, p1, path=plot_path, width=7, height=7, unit="in", dpi=400)
}

plot_bloom <- function(data_dir, file_name = "vertdd_combined_CMPOP_", version, 
                       plot_path, output_name, x_limits = c(45, 150)){
  output_name = paste0(output_name, "_", version, ".png")
  filename <- paste0(data_dir, file_name, version, ".rds")
  print ("line 87, plot_bloom")
  data <- data.table(readRDS(filename))
  data$CountyGroup = as.character(data$CountyGroup)
  data[CountyGroup == 1]$CountyGroup = 'Cooler Areas'
  data[CountyGroup == 2]$CountyGroup = 'Warmer Areas'
  print ("line 92, plot_bloom")
  d1 = subset(data, select = c("latitude", "longitude", 
                               "CountyGroup", "ClimateGroup", "ClimateScenario", 
                               "year", "month", "day", "dayofyear", 
                               "vert_Cum_dd_F", 
                               "cripps_pink", "gala", "red_deli"))

  d1 = melt(d1, id = c("latitude", "longitude", 
                       "CountyGroup", "ClimateGroup", "ClimateScenario", 
                       "year", "month", "day", "dayofyear", 
                       "vert_Cum_dd_F"))
  print ("line 103, plot_bloom")
  d1[variable == "red_deli"]$variable = "Red Delicious"
  d1[variable == "gala"]$variable = "Gala"
  d1[variable == "cripps_pink"]$variable = "Cripps Pink"
  print ("line 107, plot_bloom")
  color_ord = c("grey70", "dodgerblue", "olivedrab4", "red")
  p1 = ggplot(d1, aes(x=dayofyear, y=value, fill=factor(ClimateGroup))) +
       labs(x = "day of year", y = "proportion completing full bloom", color = "Climate Group") +
       guides(fill=guide_legend(title="Time period")) + 
     # stat_summary(geom="ribbon", fun.y=function(z) { quantile(z,0.5) }, fun.ymin=function(z) { quantile(z,0.1) }, fun.ymax=function(z) { quantile(z,0.9) }, alpha=0.3) +
     # stat_summary(geom="ribbon", fun.y=function(z) { quantile(z,0.5) }, fun.ymin=function(z) { quantile(z,0.25) }, fun.ymax=function(z) { quantile(z,0.75) }, alpha=0.7) +
       stat_summary(geom="line", fun.y=function(z) { quantile(z,0.5) }, aes(color=factor(ClimateGroup)), size = 1)+ #, aes(color=factor(Timeframe))) + , # aes(color=factor(ClimateGroup))
       scale_color_manual(values=color_ord) +
       facet_grid(. ~ variable ~ CountyGroup, scales = "free") +
     # xlim(45, 165) +
       scale_x_continuous(breaks=seq(x_limits[1], x_limits[2], 10), limits = x_limits) +
       theme(panel.grid.major = element_line(size=0.2),
             panel.spacing=unit(.5, "cm"),
             legend.title = element_blank(),
             strip.text = element_text(face="bold", size=16, color="black"),
             legend.text = element_text(size=10),
             legend.position = "bottom",
             axis.ticks = element_line(color = "black", size = .2),
             axis.text = element_text(face="plain", size=10, color="black"),
             axis.title.x = element_text(face="bold", size=16, margin=margin(t=10, r=0, b=0, l=0)),
             axis.title.y = element_text(face="bold", size=16, margin=margin(t=0, r=10, b=0, l=0))
            )
  ggsave(output_name, p1, path=plot_path, dpi=500)
}
###############################################################
#######
###############################################################
plot_cumdd_eggHatch <- function(input_dir, file_name ="combined_CMPOP_", 
                                version, output_dir, output_type="cumdd"){
  out_name = paste0(output_type, "_", version ,".png")
  #############################################
  ###    Egg Hatch
  #############################################
  if (output_type == "eggHatch"){
    # data = compute_cumdd_eggHatch(input_dir=data_dir, file_name="combined_CMPOP_", version)
    # saveRDS(data, paste0(output_dir, "/", "eggHatch_", version, ".rds"))
    data <- data.table(readRDS(paste0(output_dir, "/", "eggHatch_", version, ".rds")))
    if (version == "rcp45"){title <- "RCP 4.5"} else {title <- "RCP 8.5"}

    data$CountyGroup = as.character(data$CountyGroup)
    data[CountyGroup == 1]$CountyGroup = 'Cooler Areas'
    data[CountyGroup == 2]$CountyGroup = 'Warmer Areas'

    old_ClimateGroup <- c("Historical", "2040's", "2060's", "2080's")
    new_ClimateGroup <- c("Historical", "2040s", "2060s", "2080s")
    
    data[data$ClimateGroup == old_ClimateGroup[2]]$ClimateGroup = new_ClimateGroup[2]
    data[data$ClimateGroup == old_ClimateGroup[3]]$ClimateGroup = new_ClimateGroup[3]
    data[data$ClimateGroup == old_ClimateGroup[4]]$ClimateGroup = new_ClimateGroup[4]
    data$ClimateGroup = factor(data$ClimateGroup, levels=new_ClimateGroup, order=T)


    data = melt(data, id = c("ClimateGroup", "CountyGroup", 
                             "latitude", "longitude", 
                             "ClimateScenario", "year", "dayofyear"))
    color_ord = c("grey70", "dodgerblue", "olivedrab4", "red")
    plot = ggplot(data[value >=0.01 & value <.98 & dayofyear <360], 
                  aes(x=dayofyear, y=value, fill=factor(variable))) +
           labs(x = "day of year", y = "cumulative population fraction", fill = "larva generation") +
           facet_grid(. ~ ClimateGroup ~ CountyGroup, scales="free") +
          # geom_line(aes(fill=factor(Timeframe), color=factor(Timeframe) )) +
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
           scale_color_manual(values = color_ord,
                              labels = c("Gen. 1", "Gen. 2", "Gen. 3", "Gen. 4")) +
           scale_fill_manual(values= color_ord,
                             labels = c("Gen. 1", "Gen. 2", "Gen. 3", "Gen. 4"))+
           scale_x_continuous(limits = c(0, 320), breaks=seq(0, 300, 100)) +
           geom_vline(xintercept=c(100, 150, 200, 250, 300), linetype="solid", color ="grey", size=0.2) +
           geom_hline(yintercept=c(.25, .5, .75), linetype="solid", color ="grey", size=0.2) +
           geom_vline(xintercept=c(120, 226), linetype="solid", color ="red") +
           ggtitle(label=title) + 
           theme(plot.title = element_text(size=16, face="bold"),,
                  panel.grid.major = element_blank(),
                  panel.grid.minor = element_blank(),
                  panel.spacing=unit(.25, "cm"),
                  legend.title = element_text(face="plain", size=16),
                  legend.text = element_text(size=14),
                  legend.position = "bottom",
                  legend.key.size = unit(.65, "cm"),
                  strip.text = element_text(size=16, face="bold", color="black"),
                  axis.text = element_text(face="bold", size=14, color="black"),
                  axis.ticks = element_line(color = "black", size = .2),
                  axis.title.x = element_text(face="bold", size=16, margin=margin(t=10, r=0, b=0, l=0), color="black"),
                  axis.title.y = element_text(face="bold", size=16, margin=margin(t=0, r=10, b=0, l=0), color="black")
                  )
  return(plot)
  }
  #############################################
  ### cumdd
  #############################################
  if (output_type == "cumdd"){
    filename = paste0(input_dir, file_name, version, ".rds")
    data <- data.table(readRDS(filename))
    data = subset(data, select = c("ClimateGroup", "CountyGroup", 
                                   "latitude", "longitude", 
                                   "ClimateScenario", 
                                   "year", "dayofyear", "CumDDinF"))

    data$CountyGroup = as.character(data$CountyGroup)
    data[CountyGroup == 1]$CountyGroup = 'Cooler subregions'
    data[CountyGroup == 2]$CountyGroup = 'Warmer subregions'

    old_ClimateGroup <- c("Historical", "2040's", "2060's", "2080's")
    new_ClimateGroup <- c("Historical", "2040s", "2060s", "2080s")
    
    data[data$ClimateGroup == old_ClimateGroup[2]]$ClimateGroup = new_ClimateGroup[2]
    data[data$ClimateGroup == old_ClimateGroup[3]]$ClimateGroup = new_ClimateGroup[3]
    data[data$ClimateGroup == old_ClimateGroup[4]]$ClimateGroup = new_ClimateGroup[4]
    data$ClimateGroup = factor(data$ClimateGroup, levels=new_ClimateGroup, order=T)
    
    data <- data[, .(CumDD = median(CumDDinF)), 
                  by = c("CountyGroup", "latitude", "longitude", 
                        "ClimateScenario", "ClimateGroup", "dayofyear")]
    if (version == "rcp85"){
      y_range = seq(0, 5750, 500)
     } else {
      y_range=seq(0, 4500, 500)
    }
    color_ord = c("grey70", "dodgerblue", "olivedrab4", "red")
    plot = ggplot(data, aes(x=dayofyear, y=CumDD, fill=factor(ClimateGroup))) +
           labs(x = "day of year", y = "cumulative degree days (in F)", fill = "Climate group") +
           guides(fill=guide_legend(title="")) + 
           facet_grid(. ~ CountyGroup, scales="free") +
           #geom_line(aes(fill=factor(Timeframe), color=factor(Timeframe) )) +
           stat_summary(geom="ribbon", fun.y=function(z) { quantile(z,0.5) }, 
                                       fun.ymin=function(z) { quantile(z,0.1) }, 
                                       fun.ymax=function(z) { quantile(z,0.9) }, alpha=0.4) +
           stat_summary(geom="ribbon", fun.y=function(z) { quantile(z,0.5) }, 
                                       fun.ymin=function(z) { quantile(z,0.25) }, 
                                       fun.ymax=function(z) { quantile(z,0.75) }, alpha=0.8) + 
           stat_summary(geom="line", fun.y=function(z) { quantile(z,0.5) })+ #, aes(color=factor(Timeframe))) + 
        
           scale_color_manual(values=color_ord)+
           scale_fill_manual(values=color_ord)+
           scale_x_continuous(breaks=seq(0, 370, 50)) +
           scale_y_continuous(breaks=y_range) +
           # geom_vline(xintercept=c(90, 181, 273), 
           #           linetype="solid", 
           #           color ="red", size=0.2
           #           )+
           theme(plot.title = element_text(size=16, face="bold"),,
                panel.grid.major = element_blank(),
                panel.grid.minor = element_blank(),
                panel.spacing=unit(.25, "cm"),
                legend.title = element_text(face="plain", size=14),
                legend.text = element_text(size=12),
                legend.position = "bottom",
                legend.key.size = unit(1, "cm"), 
                strip.text = element_text(size=12, face="bold"),
                axis.text = element_text(face="bold", size=10, color="black"),
                axis.ticks = element_line(color = "black", size = .2),
                axis.title.x = element_text(face="bold", size=16, margin=margin(t=10, r=0, b=0, l=0), color="black"),
                axis.title.y = element_text(face="bold", size=16, margin=margin(t=0, r=10, b=0, l=0), color="black")
                )
    out_name = paste0(output_type, "_", version, ".png")
    return(plot)
    ggsave(out_name, plot, path=output_dir, dpi=500)
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
    color_ord = c("grey70", "dodgerblue", "olivedrab4", "red")
    plot = ggplot(data, aes(x=dayofyear, y=CumDD, fill=factor(ClimateGroup))) +
         # geom_line(aes(fill=factor(Timeframe), color=factor(Timeframe) )) +
           stat_summary(geom="ribbon", fun.y=function(z) { quantile(z,0.5) }, 
                                       fun.ymin=function(z) { quantile(z,0.1) }, 
                                       fun.ymax=function(z) { quantile(z,0.9) }, alpha=0.4) +
           stat_summary(geom="ribbon", fun.y=function(z) { quantile(z,0.5) }, 
                                       fun.ymin=function(z) { quantile(z,0.25) }, 
                                       fun.ymax=function(z) { quantile(z,0.75) }, alpha=0.8) + 
           stat_summary(geom="line", fun.y=function(z) { quantile(z,0.5) })+ #, aes(color=factor(Timeframe))) +        
           scale_color_manual(values = color_ord) +
           scale_fill_manual(values = color_ord) +
           facet_grid(CountyGroup ~ ClimateGroup ~ ., scales = "fixed") +
           scale_x_continuous(breaks=seq(0, 370, 50)) +
           scale_y_continuous(breaks=seq(0, 5000, 1000)) +
           labs(x = "day of year", y = "cumulative degree days (in F)", fill = "Climate Group") +
           theme(panel.grid.major = element_line(size = 0.7),
                 panel.spacing=unit(.5, "cm"),
                 legend.title = element_text(face = "plain", size = 16),
                 legend.text = element_text(size = 12),
                 legend.position = "bottom",
                 strip.text = element_text(size= 12, face = "plain"),
                 axis.text = element_text(face="plain", size = 10, color="black"),
                 axis.ticks = element_line(color = "black", size = .2),
                 axis.title.x = element_text(face= "plain", size = 16, margin = margin(t = 10, r = 0, b = 0, l = 0)),
                 axis.title.y = element_text(face= "plain", size = 16, margin = margin(t = 0, r = 10, b = 0, l = 0)))
  out_name = paste0("cumdd_" , version, "_type", output_type ,".png")
  ggsave(out_name, plot, path=output_dir, dpi=500)
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
  if (pop_type == "abs"){plot_abs_diapause(input_dir, file_name_extension, version, plot_path)
  } else {plot_rel_diapause(input_dir, file_name_extension, version, plot_path)}
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
               labs(x = "cumulative degree days (in F)", y = "absolute population", color = "absolute population") +
               geom_vline(xintercept=c(213, 1153, 2313, 3443, 4453), linetype="solid", color ="grey", size=.25) +
               geom_hline(yintercept=c(25, 50, 75, 100), linetype="solid", color ="grey", size=.25) +
               annotate(geom="text", x=700,  y=85, label="Gen. 1", color="black", angle=30) +
               annotate(geom="text", x=1700, y=80, label="Gen. 2", color="black", angle=30) + 
               annotate(geom="text", x=2900, y=75, label="Gen. 3", color="black", angle=30) + 
               annotate(geom="text", x=3920, y=70, label="Gen. 4", color="black", angle=30) +
               facet_grid(. ~ CountyGroup ~ ClimateGroup, scales = "free") +
               theme(panel.spacing=unit(.5, "cm"),
                     panel.grid.major = element_blank(),
                     panel.grid.minor = element_blank(),
                     axis.text = element_text(face= "plain", size = 8, color="black"),
                     axis.ticks = element_line(color = "black", size = .2),
                     axis.title.x = element_text(face= "plain", size = 12, margin = margin(t=10, r = 0, b = 0, l = 0)),
                     axis.title.y = element_text(face= "plain", size = 12, margin = margin(t=0, r = 10, b = 0, l = 0)),
                     legend.position="bottom"
                    ) + 
               scale_fill_manual(labels = c("Total", "Escape diapause"), values=c("grey", "orange"), name = "absolute population") +
               scale_color_manual(labels = c("Total", "Escape diapause"), 
                                  values=c("grey", "orange"), guide = FALSE) +
               stat_summary(geom="ribbon", fun.y=function(z) { quantile(z,0.5) }, 
                             fun.ymin=function(z) { 0 }, 
                             fun.ymax=function(z) { quantile(z,0.9) }, alpha=0.7)+
               scale_x_continuous(limits = c(0, 5000)) + 
               scale_y_continuous(limits = c(0, 100))
  
  plot_name = paste0("diapause_abs_", version, ".png")
  ggsave(plot_name, diap_plot, device="png", path=plot_path, width=10, height=7, 
         unit="in", dpi=500)
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
       labs(x = "cumulative degree days(in F)", y = "relative population", color = "relative population") +
       geom_vline(xintercept=c(213, 1153, 2313, 3443, 4453), linetype="solid", color ="grey", size=.25) +
       geom_hline(yintercept=c(5, 10, 15, 20), linetype="solid", color ="grey", size=.25) +
       annotate(geom="text", x=700,  y=18, label="Gen. 1", color="black", angle=30) +
       annotate(geom="text", x=1700, y=16, label="Gen. 2", color="black", angle=30) + 
       annotate(geom="text", x=2900, y=14, label="Gen. 3", color="black", angle=30) + 
       annotate(geom="text", x=3920, y=12, label="Gen. 4", color="black", angle=30) +
       facet_grid(. ~ CountyGroup ~ ClimateGroup, scales = "free") +
       theme(panel.grid.major = element_blank(),
             panel.grid.minor = element_blank(),
             panel.spacing=unit(.5, "cm"),
             axis.text = element_text(face= "plain", size = 8, color="black"),
             axis.ticks = element_line(color = "black", size = .2),
             axis.title.x = element_text(face= "plain", size = 12, margin = margin(t=10, r = 0, b = 0, l = 0)),
             axis.title.y = element_text(face= "plain", size = 12, margin = margin(t=0, r = 10, b = 0, l = 0)),
             legend.position="bottom"
       ) + 
       scale_fill_manual(labels = c("Total", "Escape diapause"), 
                         values=c("grey", "orange"), 
                         name = "relative population") +
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
  plot_name = paste0("diapause_rel_", version,".png")
  ggsave(plot_name, pp, device="png", path=plot_path, width=10, height=7, unit="in", dpi=500)
}
############################################################################################################
plot_adult_DoY_filling_median <- function(input_dir, file_name ="combined_CMPOP_", 
                                          version, output_dir){
  out_name = paste0("plot_Adult_DoY_median_", version ,".png")
  #############################################
  ###    Adult Emergence
  #############################################
  # data = compute_cumdd_adult_emergence_median(input_dir=data_dir, 
  #                                             file_name="combined_CMPOP_", 
  #                                             version = version)

  # saveRDS(data, paste0(output_dir, "/", "adult_DoY_filling_median_", version, ".rds"))

  data <- data.table(readRDS(paste0(output_dir, "/", "adult_DoY_filling_median_", version, ".rds")))
  if (version=="rcp45"){title <- "RCP 4.5"} else {title <- "RCP 8.5"}

  data$CountyGroup = as.character(data$CountyGroup)
  data[CountyGroup == 1]$CountyGroup = 'Cooler Areas'
  data[CountyGroup == 2]$CountyGroup = 'Warmer Areas'

  old_ClimateGroup <- c("Historical", "2040's", "2060's", "2080's")
  new_ClimateGroup <- c("Historical", "2040s", "2060s", "2080s")
  
  data[data$ClimateGroup == old_ClimateGroup[2]]$ClimateGroup = new_ClimateGroup[2]
  data[data$ClimateGroup == old_ClimateGroup[3]]$ClimateGroup = new_ClimateGroup[3]
  data[data$ClimateGroup == old_ClimateGroup[4]]$ClimateGroup = new_ClimateGroup[4]
  data$ClimateGroup = factor(data$ClimateGroup, levels=new_ClimateGroup, order=T)

  data = melt(data, id = c("ClimateGroup", "CountyGroup", 
                           "latitude", "longitude", 
                           "ClimateScenario", "year", "dayofyear"))

  color_ord = c("grey70", "dodgerblue", "olivedrab4", "red")
  plot = ggplot(data[value >=0.01 & value <.98 & dayofyear <360], 
                aes(x=dayofyear, y=value, fill=factor(variable))
                ) +
         labs(x = "day of year", y = paste0("cumulative population fraction"), fill = "adult generation") +
        # geom_line(aes(fill=factor(Timeframe), color=factor(Timeframe) )) +
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
        scale_color_manual(values = color_ord,
                          labels = c("Gen. 1", "Gen. 2", "Gen. 3", "Gen. 4")) +
        scale_fill_manual(values = color_ord,
                          labels = c("Gen. 1", "Gen. 2", "Gen. 3", "Gen. 4")) +
        facet_grid(. ~ ClimateGroup ~ CountyGroup, scales="free") +
        scale_x_continuous(limits = c(0, 320), breaks=seq(0, 300, 100)) +
        geom_vline(xintercept=c(100, 150, 200, 250, 300), linetype="solid", color ="grey", size=0.2) +
        geom_hline(yintercept=c(.25, .5, .75), linetype="solid", color ="grey", size=0.2) +
        ggtitle(label = title) + 
        theme(plot.title = element_text(size=16, face="bold"),,
              panel.grid.major = element_blank(),
              panel.grid.minor = element_blank(),
              panel.spacing=unit(.25, "cm"),
              legend.spacing.x = unit(.2, 'cm'),
              legend.title = element_text(face="plain", size=16),
              legend.text = element_text(size=14),
              legend.position = "bottom",
              legend.key.size = unit(.65, "cm"),
              strip.text = element_text(size=16, face="bold", color="black"),
              axis.text = element_text(face="bold", size=14, color="black"),
              axis.ticks = element_line(color = "black", size = .2),
              axis.title.x = element_text(face="bold", size=16, margin=margin(t=10, r=0, b=0, l=0), color="black"),
              axis.title.y = element_text(face="bold", size=16, margin=margin(t=0, r=10, b=0, l=0), color="black")
              )
  return(plot)
}

plot_adult_DoY_filling_mean <- function(input_dir, file_name ="combined_CMPOP_", 
                                        version, output_dir){
  out_name = paste0("plot_Adult_DoY_mean_", version ,".png")
  #############################################
  ###    Adult Emergence
  #############################################
  # data = compute_cumdd_adult_emergence_mean(input_dir=data_dir, 
  #                                           file_name="combined_CMPOP_", 
  #                                           version = version)
  # saveRDS(data, paste0(output_dir, "/", "adult_DoY_filling_mean_", version, ".rds"))
  data <- data.table(readRDS(paste0(output_dir, "/", "adult_DoY_filling_mean_", version, ".rds")))
  if (version=="rcp45"){title <- "RCP 4.5"} else {title <- "RCP 8.5"}

  data$CountyGroup = as.character(data$CountyGroup)
  data[CountyGroup == 1]$CountyGroup = 'Cooler Areas'
  data[CountyGroup == 2]$CountyGroup = 'Warmer Areas'

  old_ClimateGroup <- c("Historical", "2040's", "2060's", "2080's")
  new_ClimateGroup <- c("Historical", "2040s", "2060s", "2080s")
  
  data[data$ClimateGroup == old_ClimateGroup[2]]$ClimateGroup = new_ClimateGroup[2]
  data[data$ClimateGroup == old_ClimateGroup[3]]$ClimateGroup = new_ClimateGroup[3]
  data[data$ClimateGroup == old_ClimateGroup[4]]$ClimateGroup = new_ClimateGroup[4]
  data$ClimateGroup = factor(data$ClimateGroup, levels=new_ClimateGroup, order=T)

  data = melt(data, id = c("ClimateGroup", "CountyGroup", 
                           "latitude", "longitude", 
                           "ClimateScenario", "year", "dayofyear"))
  color_ord = c("grey70", "dodgerblue", "olivedrab4", "red")
  plot = ggplot(data[value >=0.01 & value <.98 & dayofyear <360], 
                aes(x=dayofyear, y=value, fill=factor(variable))
                ) +
         labs(x = "day of year", y = paste0("cumulative population fraction"), fill = "adult generation") +
         # geom_line(aes(fill=factor(Timeframe), color=factor(Timeframe) )) +
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
         scale_color_manual(values=color_ord,
                            labels = c("Gen. 1", "Gen. 2", "Gen. 3", "Gen. 4")
                            ) +
         scale_fill_manual(values=color_ord,
                           labels = c("Gen. 1", "Gen. 2", "Gen. 3", "Gen. 4")
                           ) +
         facet_grid(. ~ ClimateGroup ~ CountyGroup, scales="free") +
         scale_x_continuous(limits = c(0, 320), breaks=seq(0, 300, 100)) +
         geom_vline(xintercept=c(100, 150, 200, 250, 300, 350), 
                    linetype="solid", 
                    color ="grey", size=0.2) +
         geom_hline(yintercept=c(.25, .5, .75), linetype="solid", color ="grey", size=0.2) +
         ggtitle(label = title) + 
         theme(plot.title = element_text(size=16, face="bold"),,
              panel.grid.major = element_blank(),
              panel.grid.minor = element_blank(),
              panel.spacing=unit(.25, "cm"),
              legend.title = element_text(face="plain", size=16),
              legend.text = element_text(size=14),
              legend.position = "bottom",
              legend.spacing.x = unit(.2, 'cm'),
              legend.key.size = unit(.65, "cm"), 
              strip.text = element_text(size=16, face="bold", color="black"),
              axis.text = element_text(face="bold", size=14, color="black"),
              axis.ticks = element_line(color = "black", size = .2),
              axis.title.x = element_text(face="bold", size=16, margin=margin(t=10, r=0, b=0, l=0), color="black"),
              axis.title.y = element_text(face="bold", size=16, margin=margin(t=0, r=10, b=0, l=0), color="black")
              )

  return(plot)
  }
#####################################################################################
#######################                                   ###########################
#######################          Sensitivity Plots        ###########################
#######################                                   ###########################
#####################################################################################
plot_scale_sensitivity_dot <- function(master_path, numeric_shifts){
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
          mask_entry = paste0(model, stag, dead, "_")
          mask = c(paste0(mask_entry, "2040"), paste0(mask_entry, "2060"),
                   paste0(mask_entry, "2080"))
          
          curr_data = all_info[all_info$pop_type %in% mask]
          
          legend_labels = c("2040s", "2060s", "2080s")
          
          h_line_coord = as.numeric(all_info[all_info$pop_type %in% c(paste0(mask_entry, "historical"))][1, 3])
          history_line <- data.frame( x = c(-Inf, Inf), y = h_line_coord, history_line = factor(h_line_coord) )
          dot_plot =ggplot(curr_data, aes(x=shifts*100, y=generation, color=pop_type)) + 
                    geom_point() +
                    geom_line() + 
                    geom_line(aes(x, y, linetype = "Historical" ), history_line, inherit.aes = FALSE) +
                    ylim(1.5, 4) + 
                    theme(panel.grid.major = element_line(size = 0.3),
                          panel.grid.minor = element_line(size = 0.2),
                          panel.spacing=unit(.5, "cm"),
                          legend.position="bottom",
                          legend.title = element_blank(),
                          legend.text = element_text(size=10, face="plain"),
                          legend.spacing.x = unit(.05, 'cm'),
                          legend.key.size = unit(.5, "cm"),
                          legend.margin=margin(t= -.5, r = 0, b = 0, l = 0),
                          axis.ticks = element_line(color = "black", size = .2),
                          axis.title.x = element_text(face = "plain", size=12, margin = margin(t=10, r=0, b=0, l=0)),
                          axis.title.y = element_text(face = "plain", size=12, margin = margin(t=0, r=10, b=0, l=0))
                    ) + 
                    scale_color_discrete(breaks=mask, labels= legend_labels) +
                    labs(x="Weibull scale parameter change by %", y="number of generations")
                  plot_name = paste0(file_pref, model, stag, dead, "_scale_sens", weather, ".png")
                  ggsave(plot_name, dot_plot, path=master_path, device="png", 
                         dpi=500, width=5.57, height=5.42, unit="in")
        }
      }
    }
  }
}
#################################################################################################
##################                                                             ##################
##################                    Box Plots                                ##################
##################                                                             ##################
#################################################################################################
plot_adult_emergence_4_Latex <- function(input_dir, file_name, 
                                         box_width=.2, plot_path, output_name, 
                                         color_ord = c("grey70", "dodgerblue", "olivedrab4", "red"),
                                         plot_width=8,
                                         plot_height=6){
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
  # df <- data.frame(data)
  # df <- (df %>% group_by(CountyGroup, ClimateGroup))
  # medians <- (df %>% summarise(med = median(Emergence)))
  # medians_vec <- medians$med

  df <- data.frame(data) %>% 
        group_by(CountyGroup, ClimateGroup)) %>% 
        summarise(med = median(Emergence))
  medians_vec <- medians$med
  
  p = ggplot(data = data, aes(x=ClimateGroup, y=Emergence, fill=ClimateGroup))+
      geom_boxplot(outlier.size=-.25, notch=TRUE, width=box_width, lwd=.1) +
      scale_x_discrete(expand=c(0, 2), limits = levels(data$ClimateGroup[1])) +
      scale_y_continuous(breaks = round(seq(40, 170, by = 20))) +
      labs(x="", y="day of year", color = "Climate group") +
      facet_wrap(~CountyGroup) +
      theme(plot.margin = unit(c(t=0, r=.2, b=.1, l=0), "cm"),
            panel.border = element_rect(fill=NA, size=.3),
            panel.grid.major = element_line(size = 0.05),
            panel.grid.minor = element_blank(),
            panel.spacing=unit(.5, "cm"),
            panel.spacing = unit(.25, "cm"),
            legend.position = "bottom", 
            legend.key.size = unit(.75, "line"),
            legend.text=element_text(size=5),
            legend.margin=margin(t= -.3, r = 0, b = 0, l = 0, unit = 'cm'),
            legend.spacing.x = unit(.05, 'cm'),
            legend.title = element_blank(),
            strip.text.x = element_text(size = 5),
            axis.ticks = element_line(color = "black", size = .2),
            axis.text = element_text(face = "plain", size = 2.5, color = "black"),
            axis.title.x = element_text(face = "plain", size=6, margin = margin(t=2.5, r=0, b=0, l=0)),
            axis.text.x = element_text(size = 4, face="plain", color="black"),
            axis.title.y = element_text(face = "plain", size=6, margin = margin(t=0, r=0, b=0, l=0)),
            axis.text.y  = element_blank(),
            axis.ticks.y = element_blank()
        ) +
    scale_fill_manual(values=color_ord,
                      name="Time\nPeriod", 
                      labels=c("Historical", "2040s","2060s","2080s")) + 
    scale_color_manual(values=color_ord,
                       name="Time\nPeriod", 
                       limits = color_ord,
                       labels=c("Historical","2040s","2060s","2080s")) + 
    geom_text(data = medians, 
              aes(label = sprintf("%1.0f", medians$med), y=medians$med), 
              size=1.3, 
              position =  position_dodge(.09),
              vjust = -1.4) +
    coord_flip()
  ggsave(output_name, p, path=plot_path, device="png", 
                         width=plot_width, height=plot_height, unit="cm", dpi=500)
}
##################
##################   Adult Emergence
##################
plot_adult_emergence <- function(input_dir, file_name, 
                                 box_width=0.7, plot_path, output_name, 
                                 color_ord = c("grey47", "dodgerblue", "olivedrab4", "red")
                                 ){

  if (em=="rcp45"){plot_title <- "RCP 4.5"} else {plot_title <- "RCP 8.5"}
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
  df <- df %>% group_by(CountyGroup, ClimateGroup)

  medians <- df %>% summarise(med = median(Emergence))
  medians_vec <- medians$med

  the_theme<- theme(plot.title = element_text(size=30, face="bold"),
                   panel.grid.minor = element_blank(),
                   panel.spacing=unit(.5, "cm"),
                   legend.margin=margin(t=.5, r = 0, b = 0, l = 0, unit = 'cm'),
                   legend.title = element_blank(),
                   legend.position="bottom", 
                   legend.key.size = unit(4, "line"),
                   legend.spacing.x = unit(.05, 'cm'),
                   panel.grid.major = element_line(size = 0.1),
                   axis.ticks = element_line(color="black", size = .2),
                   strip.text = element_text(size=25, face = "bold"),
                   legend.text=element_text(size=25),
                   axis.title.x = element_text(size=25, face = "bold",  margin = margin(t=8, r=0, b=0, l=0)),    
                   axis.text.x = element_text(size= 20, face = "bold", color="black"),
                   axis.title.y = element_blank(),
                   axis.text.y  = element_blank(),
                   axis.ticks.y = element_blank()
                  ) 
  
  p = ggplot(data = data, aes(x=ClimateGroup, y=Emergence, fill=ClimateGroup))+
      geom_boxplot(outlier.size=-.15, notch=FALSE, width=box_width, lwd=.25) +
      scale_x_discrete(expand=c(0, 2), limits = levels(data$ClimateGroup[1])) +
      scale_y_continuous(breaks = round(seq(40, 170, by = 20), 1)) +
      labs(x="Time period", y="day of year of adult emergence", color = "Climate Group") +
      facet_wrap(~CountyGroup) +
      the_theme +
      scale_fill_manual(values=color_ord, name="Time\nPeriod", labels=time_periods) + 
      scale_color_manual(values=color_ord, name="Time\nPeriod", limits = color_ord, labels= time_periods) +
      # geom_text(data = medians, 
      #           aes(label = sprintf("%1.0f", medians$med), y=medians$med), 
      #               size=4, position=position_dodge(1), vjust = -1) +
      coord_flip() + 
      ggtitle(label = plot_title)

  return(p)
}
##################
##################   Generations of Adults or Larva by Aug 23
##################
plot_No_generations_4_latex <- function(input_dir,
                                        file_name,
                                        stage,
                                        dead_line,
                                        box_width=.25,
                                        plot_width = 6.5,
                                        plot_height = 2.5,
                                        plot_path,
                                        version,
                                        color_ord = c("grey70", "dodgerblue", "olivedrab4", "red")){
  file_name <- paste0(input_dir, file_name)
  data <- data.table(readRDS(file_name))
  data$CountyGroup = as.character(data$CountyGroup)
  data[CountyGroup == 1]$CountyGroup = 'Cooler Areas'
  data[CountyGroup == 2]$CountyGroup = 'Warmer Areas'
  
  if (stage=="Larva"){
    var = "NumLarvaGens"
  } else if (stage=="Adult"){
    var = "NumAdultGens"
  }
  
  data <- subset(data, select = c("ClimateGroup", "CountyGroup", var))
  df <- data.frame(data)
  df <- (df %>% group_by(CountyGroup, ClimateGroup))
  medians <- (df %>% summarise(med = median(!!sym(var))))
  rm(df)
  if (dead_line=="Aug"){
    y_lab = paste0("number of ", stage, " generations by Aug. 23")
  } else{
    y_lab = paste0("number of ", stage, " generations by Nov. 5")
  }
  
  box_plot = ggplot(data = data, aes(x = ClimateGroup, y = !!sym(var), fill = ClimateGroup)) + 
             geom_boxplot(outlier.size=-.35, lwd=0.1, notch=TRUE, width=box_width) +
             # The bigger the number in expand below, the smaller the space between y-ticks
             scale_x_discrete(expand=c(0, 2), limits = levels(data$ClimateGroup[1])) +
             scale_y_continuous(limits = c(.5, 4), breaks=seq(1, 5, by=1)) + 
             labs(#x="", 
                  y=y_lab, 
                  color = "Climate Group") +
             facet_wrap(~CountyGroup) +
             theme(plot.margin = unit(c(t=0.1, r=0.1, b=.1, l=-0.01), "cm"),
                   panel.border = element_rect(fill=NA, size=.3),
                   panel.grid.major = element_line(size = 0.05),
                   panel.grid.minor = element_blank(),
                   panel.spacing=unit(.5,"cm"),
                   legend.position="bottom", 
                   legend.title = element_blank(),
                   legend.key.size = unit(.75,"line"),
                   legend.text=element_text(size=5),
                   legend.margin=margin(t= -.3, r = 0, b = 0, l = 0, unit = 'cm'),
                   legend.spacing.x = unit(.05, 'cm'),
                   strip.text.x = element_text(size = 5),
                   axis.ticks = element_line(color = "black", size = .2),
                   axis.title.x = element_text(face = "plain", size=6, margin = margin(t=2.5, r=0, b=0, l=0)),
                   axis.text.x = element_text(size = 4, face="plain", color="black"),
                   axis.title.y = element_blank(),
                   axis.text.y  = element_blank(),
                   axis.ticks.y = element_blank()
             ) +
           scale_fill_manual(values=color_ord,
                             name="Time\nPeriod", 
                             labels=c("Historical","2040s","2060s","2080s")) + 
          scale_color_manual(values=color_ord,
                             name="Time\nPeriod", 
                             limits = color_ord,
                             labels=c("Historical","2040s","2060s","2080s")) + 
          geom_text(data = medians, 
                    aes(label = sprintf("%1.1f", medians$med), y=medians$med), 
                    size=1.3, 
                    position =  position_dodge(.09),
                    vjust = -1.4) +
            coord_flip()
  
  plot_name = paste0(stage, "_Gen_", dead_line, "_", version, ".png")
  ggsave(plot_name, 
         box_plot, 
         path=plot_path, 
         device="png", 
         width=plot_width, height=plot_height, units = "cm", dpi=500)
}

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
    y_lab = paste0("number of ", tolower(stage), " generations by Aug. 23")
  } else{
    y_lab = paste0("number of ", tolower(stage), " generations by Nov. 5")
  }
  box_plot = ggplot(data = data, aes(x = ClimateGroup, y = !!sym(var), fill = ClimateGroup)) + 
             geom_boxplot(outlier.size=-.15, lwd=0.25, notch=TRUE, width=box_width) +
             # The bigger the number in expand below, the smaller the space between y-ticks
             scale_x_discrete(expand=c(0, 3), limits = levels(data$ClimateGroup[1])) +
             scale_y_continuous(limits = c(.5, 4), breaks=seq(1, 5, by=1)) + 
             labs(x="Time period", 
                  y=y_lab, 
                  color = "Climate Group") +
             facet_wrap(~CountyGroup) +
             theme(legend.position="bottom", 
             	     legend.key.size = unit(.75, "line"),
                   legend.text=element_text(size=5),
                   legend.margin=margin(t=-.1, r=0, b=0, l=0, unit='cm'),
                   legend.spacing.x = unit(.05, 'cm'),
                   legend.title = element_blank(),
                   panel.grid.minor = element_blank(),
                   panel.spacing=unit(.5, "cm"),
                   axis.text = element_text(face = "plain", size = 10, color="black"),
                   axis.ticks = element_line(color = "black", size = .2),
                   axis.text.x = element_text(size = 7, color="black"),
                   axis.title.x = element_text(face = "plain", size=8, 
                                               margin = margin(t=3, r=0, b=0, l=0)),
                  
                   axis.title.y = element_text(face = "plain", size=8, 
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
         width=plot_with, height=plot_height, 
         units = "in", dpi=500)
}
##################
################## Flight vs. DoY
##################
plot_flight_DoY_half <- function(input_dir, input_name, stage, 
                                 output_dir, output_name, 
                                 plot_with=7, plot_height=3){
  ##
  ## This function plots the box plot for when the
  ## population of each generation of adult/larva hits 50%.
  ##
  color_ord <- c("grey47", "dodgerblue", "olivedrab4", "red")
  data <- readRDS(paste0(input_dir, input_name))

  if (stage == "adult"){
    data <- subset(data, select = c("AGen1_0.5", "AGen2_0.5", "AGen3_0.5", "AGen4_0.5",
                                    "ClimateGroup", "CountyGroup"))
    L = c('AGen1_0.5','AGen2_0.5', 'AGen3_0.5','AGen4_0.5')
  } else{
    data <- subset(data, select = c("LGen1_0.5", "LGen2_0.5", "LGen3_0.5", "LGen4_0.5",
                                    "ClimateGroup", "CountyGroup"))
    L = c('LGen1_0.5', 'LGen2_0.5',  'LGen3_0.5', 'LGen4_0.5')
  }
  
  data$CountyGroup = as.character(data$CountyGroup)
  data[CountyGroup == 1]$CountyGroup = 'Cooler Areas'
  data[CountyGroup == 2]$CountyGroup = 'Warmer Areas'
  
  data_melted = melt(data, id = c("ClimateGroup", "CountyGroup"))
  data_melted$variable <- factor(data_melted$variable, levels = L, ordered = TRUE)
  rm(data)
  bplot <- ggplot(data = data_melted, aes(x=variable, y=value), group = variable) + 
           geom_boxplot(outlier.size=-.15, notch=FALSE, width=.4, lwd=.25, aes(fill=ClimateGroup), 
                        position=position_dodge(width=0.5)) + 
           scale_y_continuous(limits = c(80, 370), breaks = seq(100, 360, by = 50)) +
           facet_wrap(~CountyGroup, scales="free", ncol=6, dir="v") + 
           labs(x="Time period", y="day of year", color = "Climate Group") + 
           theme(legend.position="bottom", 
                 legend.margin=margin(t=-.1, r=0, b=5, l=0, unit = 'cm'),
                 legend.title = element_blank(),
                 legend.text = element_text(size=7, face="plain"),
                 legend.key.size = unit(.5, "cm"), 
                 panel.grid.major = element_line(size = 0.1),
                 panel.grid.minor = element_line(size = 0.1),
                 panel.spacing=unit(.5, "cm"),
                 strip.text = element_text(size= 6, face = "plain"),
                 axis.text = element_text(face = "plain", size = 4, color="black"),
                 axis.ticks = element_line(color = "black", size = .2),
                 axis.title.x = element_text(face = "plain", size = 10, 
                                             margin = margin(t=10, r=0, b=0, l=0)),
                 axis.text.x = element_text(size = 6, color="black"), # tick text font size
                 axis.title.y = element_text(face = "plain", size = 10, 
                                             margin = margin(t=0, r=7, b=0, l=0)),
                 axis.text.y  = element_blank(),
                 axis.ticks.y = element_blank(),
                 plot.margin = unit(c(t=-0.35, r=.7, b=-4.7, l=0.3), "cm")
                ) +
           scale_color_manual(values=color_ord,
                              name="Time\nPeriod", 
                              limits = color_ord,
                              labels=c("Historical", "2040", "2060", "2080")) +
           scale_fill_manual(values=color_ord,
                             name="Time\nPeriod", 
                             labels=c("Historical", "2040", "2060", "2080")) + 
           coord_flip()
  #bplot <- add_sub(bplot, label="Gen. 1", x=1.02, y=8, angle=270, size=6, fontface="plain")
  ggsave(output_name, bplot, device="png", 
         path=plot_path, width=plot_with, height=plot_height, 
         unit="in", dpi=500)
}

##############################################
##################
################## Pre Approved Funcs.
##################
##############################################
plot_cumdd_seasonal <- function(input_dir, file_name ="combined_CMPOP_", 
                                version, output_dir){
  # This could have been achieved just by vertical lines!
  out_name = paste0("plot_", "_", version ,".png")
  filename = paste0(input_dir, file_name, version, ".rds")
  data <- data.table(readRDS(filename))

  data$CountyGroup = as.character(data$CountyGroup)
  data[CountyGroup == 1]$CountyGroup = 'Cooler Areas'
  data[CountyGroup == 2]$CountyGroup = 'Warmer Areas'
  data = subset(data, select = c("ClimateGroup", "CountyGroup", 
                                 "latitude", "longitude", 
                                 "ClimateScenario", 
                                 "year", "dayofyear", "CumDDinF"))
  # data$CumDD = data$CumDDinF
  data <- data[, .(CumDD = median(CumDDinF)), by = c("CountyGroup", "latitude", "longitude", 
                                                     "ClimateScenario", "ClimateGroup", "dayofyear")]
  if (version == "rcp85"){
    y_range = seq(0, 5750, 500)
  } else {
    y_range = seq(0, 4500, 500)
  }

  # add the new season column
  data[, season := as.character(0)]
  data[data[ , data$dayofyear <= 90]]$season = "Qr. 1"
  data[data[ , data$dayofyear >= 91 & data$dayofyear <= 181]]$season = "Qr. 2"
  data[data[ , data$dayofyear >= 182 & data$dayofyear <= 273]]$season = "Qr. 3"
  data[data[ , data$dayofyear >= 274]]$season = "Qr. 4"
  data$season = factor(data$season, levels = c("Qr. 1", "Qr. 2", "Qr. 3", "Qr. 4s"))
  color_ord = c("grey70", "dodgerblue", "olivedrab4", "red")
  plot = ggplot(data, aes(x=dayofyear, y=CumDD, fill=factor(ClimateGroup))) +
         facet_grid(. ~ CountyGroup ~ season, scales = "free") +
         labs(x = "day of year", y = "cumulative degree days (in F)", fill = "Climate group") +
         guides(fill=guide_legend(title="")) + 
         stat_summary(geom="ribbon", fun.y=function(z) { quantile(z,0.5) }, 
                                     fun.ymin=function(z) { quantile(z,0.1) }, 
                                     fun.ymax=function(z) { quantile(z,0.9) }, alpha=0.4) +
         stat_summary(geom="ribbon", fun.y=function(z) { quantile(z,0.5) }, 
                                     fun.ymin=function(z) { quantile(z,0.25) }, 
                                     fun.ymax=function(z) { quantile(z,0.75) }, alpha=0.8) + 
         stat_summary(geom="line", fun.y=function(z) { quantile(z,0.5) })+
      
         scale_color_manual(values=color_ord) +
         scale_fill_manual(values=color_ord)+ 
         scale_x_continuous(breaks=seq(0, 370, 50)) +
         scale_y_continuous(breaks=y_range) +
         theme(panel.grid.minor = element_blank(),
               panel.spacing=unit(.5, "cm"),
               legend.title = element_text(face = "plain", size = 12),
               legend.text = element_text(size = 10),
               legend.position = "bottom",
               strip.text = element_text(size = 12, face = "plain"),
               axis.text = element_text(face = "plain", size = 10, color="black"),
               axis.ticks = element_line(color = "black", size = .2),
               axis.title.x = element_text(face = "plain", size=16, margin=margin(t=10, r=0, b=0, l=0)),
               axis.title.y = element_text(face = "plain", size=16, margin=margin(t=0, r=10, b=0, l=0))
               )       
  out_name = paste0("cumDD_seasonal_", version, ".png")
  ggsave(out_name, plot, path=output_dir, dpi=500)
}


plot_cumdd_histogram <- function(input_dir, file_name ="combined_CMPOP_", 
                                 version, output_dir){
    filename = paste0(input_dir, file_name, version, ".rds")
    data <- data.table(readRDS(filename))

    data$CountyGroup = as.character(data$CountyGroup)
    data[CountyGroup == 1]$CountyGroup = 'Cooler Areas'
    data[CountyGroup == 2]$CountyGroup = 'Warmer Areas'
    data = subset(data, select = c("ClimateGroup", "CountyGroup", 
                                   "latitude", "longitude", 
                                   "ClimateScenario", 
                                   "year", "dayofyear", "CumDDinF"))

    data <- data[, .(CumDD = median(CumDDinF)), by = c("CountyGroup", "latitude", "longitude", 
                                                       "ClimateScenario", "ClimateGroup", "dayofyear")]
    if (version == "rcp85"){
      y_range = seq(0, 5750, 500)
    } else {
      y_range=seq(0, 4500, 500)
      }
    color_ord = c("grey70", "dodgerblue", "olivedrab4", "red")
    plot = ggplot(data, aes(x=dayofyear, fill=ClimateGroup, color=ClimateGroup)) + 
           geom_histogram(alpha=0.5, position="dodge", bins=4) +
           labs(x = "day of year", fill="") +
           guides(guide_legend(title=""), color = FALSE, shape = FALSE) + 
           facet_grid(. ~ CountyGroup, scales="free") +
           scale_x_continuous(breaks=seq(0, 370, 50)) +
           #scale_fill_discrete(name = "") +
           scale_color_manual(values=color_ord) +
           scale_fill_manual(values=color_ord) + 
           geom_vline(xintercept=c(90, 181, 273, 366), 
                      linetype="dashed", 
                      color ="black", size=0.3) +
           guides(guide_legend(title="")) + 
           theme(panel.grid.minor = element_blank(),
                 panel.spacing=unit(.5, "cm"),
                 legend.text = element_text(size = 8),
                 legend.position = "bottom",
                 legend.margin=margin(t=-.4, r=0, b=.1, l=0, unit = 'cm'),
                 strip.text = element_text(size = 10, face = "plain"),
                 axis.text = element_text(face = "plain", size = 10, color="black"),
                 axis.ticks = element_line(color = "black", size = .2),
                 axis.title.x = element_text(face = "plain", size=12, margin=margin(t=10, r=0, b=0, l=0)),
                 axis.title.y = element_blank()
                 )
    out_name = paste0("cumdd_historgam_seasonally_", version, ".png")
    ggsave(out_name, plot, width=21, height=7, unit="in", path=output_dir, dpi=500)
}


cumulative_qrt_boxplot <- function(input_dir, file_name, version, output_dir){
  # This stupid thing does not plot on Aeolus.
  # First I did subset the data to pull proper information,
  # and then I used it on my computer!
  filename = paste0(input_dir, file_name, version, ".rds")
  data <- data.table(readRDS(filename))
  data = subset(data, select = c("ClimateGroup", "CountyGroup",
                               "dayofyear", "CumDDinF"))
  data$CountyGroup = as.character(data$CountyGroup)
  data[CountyGroup == 1]$CountyGroup = 'Cooler Areas'
  data[CountyGroup == 2]$CountyGroup = 'Warmer Areas'

  # add the new season column
  data[, season := as.character(0)]
  data[data[ , data$dayofyear <= 90]]$season = "QTR 1"
  data[data[ , data$dayofyear >= 91 & data$dayofyear <= 181]]$season = "QTR 2"
  data[data[ , data$dayofyear >= 182 & data$dayofyear <= 273]]$season = "QTR 3"
  data[data[ , data$dayofyear >= 274]]$season = "QTR 4"
  data = within(data, remove(dayofyear))
  data$season = factor(data$season, levels = c("QTR 1", "QTR 2", "QTR 3", "QTR 4"))
  
  # df <- data.frame(data)
  # df <- (df %>% group_by(CountyGroup, ClimateGroup, season))
  # medians <- (df %>% summarise(med = median(CumDDinF)))
  # medians <- medians$med
  # rm(df)
  
  data = melt(data, id = c("ClimateGroup", "CountyGroup", "season"))
  data = within(data, remove(variable))
  bplot <- ggplot(data = data, aes(x=season, y=value), group = season) + 
           geom_boxplot(outlier.size=-.15, notch=FALSE, width=.4, lwd=.25, aes(fill=ClimateGroup), 
                        position=position_dodge(width=0.5)) + 
           scale_y_continuous(limits = c(0, 6000), breaks = seq(0, 6000, by = 500)) + 
           facet_wrap(~CountyGroup, scales="free", ncol=6, dir="v") + 
           labs(x="", y="cumulative degree day", color = "Climate Group") + 
           theme(legend.position="bottom", 
                 legend.margin=margin(t=-.7, r=0, b=5, l=0, unit = 'cm'),
                 legend.title = element_blank(),
                 legend.text = element_text(size=10, face="plain"),
                 legend.key.size = unit(.5, "cm"), 
                 panel.grid.major = element_line(size = 0.1),
                 panel.grid.minor = element_line(size = 0.1),
                 panel.spacing=unit(.5, "cm"),
                 strip.text = element_text(size= 10, face = "plain"),
                 axis.text = element_text(face = "plain", size = 4, color="black"),
                 axis.ticks = element_line(color = "black", size = .2),
                 axis.title.x = element_text(face = "plain", size = 10, 
                                             margin = margin(t=10, r=0, b=0, l=0)),
                 axis.text.x = element_text(size = 10, color="black"), # tick text font size
                 axis.text.y = element_text(size = 10, color="black"),
                 axis.title.y = element_text(face = "plain", size = 12, 
                                             margin = margin(t=0, r=7, b=0, l=0)),
                 plot.margin = unit(c(t=0.3, r=.7, b=-4.7, l=0.3), "cm")
                )
  out_name = paste0("cumdd_qrt_", version, ".png")
  ggsave(out_name, bplot, width=15, height=7, unit="in", path=output_dir, dpi=500, device="png")
}


#################################################################################################
##################                                                             ##################
##################                    LefLEt Maps                              ##################
##################                                                             ##################
#################################################################################################
# library(shiny)
# library(shinydashboard)
# library(htmlwidgets)
# library(webshot)
# library(shinyBS)
# library(rgdal)    # for readOGR and others
# library(maps)
# library(sp)       # for spatial objects
# library(leaflet)  # for interactive maps (NOT leafletR here)
# library(dplyr)    # for working with data frames
# library(ggplot2)  # for plotting
# library(data.table)
# library(reshape2)
# library(RColorBrewer)
##################
################## 
##################










