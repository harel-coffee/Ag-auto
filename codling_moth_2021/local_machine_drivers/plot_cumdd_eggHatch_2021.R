#
#  Feb. 18, 2021
#  This is a copy of a file with the same name from 2018.
# 
#

#
# Core and coreplot cannot be imported since Core.R has imported read_binary.R from Aeolus path.
# perhaps we can import read_binary in the drivers as opposed to core.R
#
# source_path1 = "/Users/hn/Documents/00_GitHub/Ag/codling_moth_2021/core.R"
# source_path2 = "/Users/hn/Documents/00_GitHub/Ag/codling_moth_2021/core_plot.R"
# source(source_path1)
# source(source_path2)


library(data.table)
library(ggplot2)
library(dplyr)

#
#  The reason I have this function here is that due 
#  do having aeolus path in core.R I cannot import core.R here.
#

compute_cumdd_eggHatch <- function(data){

  data <- subset(data, select = c("latitude", "longitude", "city",
                                  "model", "time_period", 
                                  "year", "dayofyear", "emission",
                                  "PercLarvaGen1", "PercLarvaGen2", 
                                  "PercLarvaGen3", "PercLarvaGen4"))
    
  data <- data[, .(LarvaGen1 = mean(PercLarvaGen1), LarvaGen2 = mean(PercLarvaGen2), 
                   LarvaGen3 = mean(PercLarvaGen3), LarvaGen4 = mean(PercLarvaGen4)), 
                   by = c("latitude", "longitude", "city", "emission",
                          "model", "time_period", "year", "dayofyear")]
  return (data)
}

plot_cumdd_eggHatch_2021 <- function(data, output_type="eggHatch"){
  #
  # The output_type == "eggHatch" is updated on Feb 18 2021. Not the other part
  #

  out_name = paste0(output_type ,".png")
  #############################################
  ###
  ###    Egg Hatch
  ###
  #############################################
  if (output_type == "eggHatch"){
    data <- compute_cumdd_eggHatch(data)
    data = melt(data, id = c("time_period", "city", 
                             "latitude", "longitude", "emission",
                             "model", "year", "dayofyear"))
    
    color_ord = c("grey47", "dodgerblue", "olivedrab4", "red")

    data <- data.table(data)
    plot <- ggplot(data[value >=0.01 & value <.98 & dayofyear<360], 
                  aes(x=dayofyear, y=value, fill=factor(variable))) +
           labs(x = "day of year", y = "cumulative population fraction", fill = "larva generation") +
           facet_grid( ~ emission ~ city, scales="free") +
          # geom_line(aes(fill=factor(Timeframe), color=factor(Timeframe) )) +
           stat_summary(geom = "ribbon", 
                        fun = function(z) { quantile(z,0.5) }, 
                        fun.min = function(z) { quantile(z,0.1) }, 
                        fun.max = function(z) { quantile(z,0.9) }, 
                        alpha = 0.3) +
           stat_summary(geom = "ribbon", 
                        fun = function(z) { quantile(z,0.5) }, 
                        fun.min = function(z) { quantile(z,0.25) }, 
                        fun.max = function(z) { quantile(z,0.75) }, 
                        alpha = 0.8) + 
           stat_summary(geom = "line", 
                        fun = function(z) { quantile(z,0.5) }) +
           scale_color_manual(values = color_ord,
                              labels = c("Gen. 1", "Gen. 2", "Gen. 3", "Gen. 4")) +

           scale_fill_manual(values= color_ord,
                             labels = c("Gen. 1", "Gen. 2", "Gen. 3", "Gen. 4")) +

           scale_x_continuous(limits = c(0, 360), breaks=seq(0, 300, 100)) +

           geom_vline(xintercept=c(100, 150, 200, 250, 300), linetype="solid", color ="grey", size=0.2) +

           geom_hline(yintercept=c(.25, .5, .75), linetype="solid", color ="grey", size=0.2) +

           geom_vline(xintercept=c(120, 226), linetype="solid", color ="red") +
           theme(plot.title = element_text(size=16, face="bold"),,
                 panel.grid.major = element_blank(),
                 panel.grid.minor = element_blank(),
                 panel.spacing=unit(.25, "cm"),
                 legend.title = element_text(face="plain", size=16),
                 legend.text = element_text(size=14),
                 legend.position = "bottom",
                 legend.key.size = unit(.65, "cm"),
                 strip.text = element_text(size=16, face="bold", color="black"),
                 axis.text = element_text(face="plain", size=14, color="black"),
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
    filename = paste0(input_dir, file_name, emission, ".rds")
    data <- data.table(readRDS(filename))
    data = subset(data, select = c("time_period", "city", 
                                   "latitude", "longitude", 
                                   "model", 
                                   "year", "dayofyear", "CumDDinF"))

    data$time_period = factor(data$time_period, levels=new_time_period, order=T)
    
    data <- data[, .(CumDD = median(CumDDinF)), 
                  by = c("city", "latitude", "longitude", 
                        "model", "time_period", "dayofyear")]
    if (emission == "rcp85"){
      y_range = seq(0, 5750, 500)
     } else {
      y_range=seq(0, 4500, 500)
    }
    color_ord = c("grey47", "dodgerblue", "olivedrab4", "red")
    plot = ggplot(data, aes(x=dayofyear, y=CumDD, fill=factor(time_period))) +
           labs(x = "day of year", y = "cumulative degree days (in F)", fill = "Climate group") +
           guides(fill=guide_legend(title="")) + 
           facet_grid(. ~ city, scales="free") +
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
    out_name = paste0(output_type, "_", emission, ".png")
    return(plot)
  }

  if (output_type==3){
    filename = paste0(input_dir, file_name, emission, ".rds")
    data <- data.table(readRDS(filename))

    data = subset(data, select = c("time_period", "city", "latitude", 
                                   "longitude", "model", "year", "dayofyear", "CumDDinF"))
    data$CumDD = data$CumDDinF
    color_ord = c("grey47", "dodgerblue", "olivedrab4", "red")
    plot = ggplot(data, aes(x=dayofyear, y=CumDD, fill=factor(time_period))) +
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
           facet_grid(city ~ time_period ~ ., scales = "fixed") +
           scale_x_continuous(breaks=seq(0, 370, 50)) +
           scale_y_continuous(breaks=seq(0, 5000, 1000)) +
           labs(x = "day of year", y = "cumulative degree days (in F)", fill = "Climate Group") +
           theme(panel.grid.major = element_blank(),
                 panel.spacing=unit(.5, "cm"),
                 legend.title = element_text(face = "plain", size = 16),
                 legend.text = element_text(size = 12),
                 legend.position = "bottom",
                 strip.text = element_text(size= 12, face = "plain"),
                 axis.text = element_text(face="plain", size = 10, color="black"),
                 axis.ticks = element_line(color = "black", size = .2),
                 axis.title.x = element_text(face= "bold", size = 16, margin = margin(t = 10, r = 0, b = 0, l = 0)),
                 axis.title.y = element_text(face= "bold", size = 16, margin = margin(t = 0, r = 10, b = 0, l = 0)))
  
  }
}

########################################################################################

data_dir <- "/Users/hn/Documents/01_research_data/codling_moth_2021/01_combined_files/"

############################################################################################
#
#   Read, clean up, and process the data for plot purposes
#

CMPOP_rcp45 <- readRDS(paste0(data_dir, "LM_combined_CMPOP_rcp45.rds"))
CMPOP_rcp85 <- readRDS(paste0(data_dir, "LM_combined_CMPOP_rcp85.rds"))

CMPOP_observed <- readRDS(paste0(data_dir, "LO_combined_CMPOP_observed.rds"))
CMPOP_observed$time_period <- "Historical"

# 
#  change emission in observed so that thet appear
#  in both emission scenario plots
#

CMPOP_observed_45 <- CMPOP_observed
CMPOP_observed_85 <- CMPOP_observed

CMPOP_observed_45$emission <- "RCP 4.5"
CMPOP_observed_85$emission <- "RCP 8.5"

data <- rbind(CMPOP_rcp45, CMPOP_rcp85, 
              CMPOP_observed_45, CMPOP_observed_85)

data$time_period <- as.character(data$time_period)
time_levels = c("Historical", "2040s", "2060s", "2080s")
data$time_period <- factor(data$time_period, levels = time_levels, ordered = TRUE)

data$emission <- factor(data$emission, levels = c("RCP 4.5", "RCP 8.5"), ordered = TRUE)


output_type = "eggHatch"

plot_path = "/Users/hn/Documents/01_research_data/codling_moth_2021/03_plots_4_paper/"

plt <- plot_cumdd_eggHatch_2021(data, output_type="eggHatch")


#
#  For vertical direction
#
# ggsave(filename = "egghatch_2021.png", 
#        path = plot_path, 
#        plot = plt,
#        height = 9, width = 7, units = "in",
#        dpi = 400, 
#        device = "png")

H = 5
W = 11
ggsave(filename = paste0("egghatch_2021_", H, "_by_", W, ".png"), 
       path = plot_path, 
       plot = plt,
       height = H, width = W, units = "in",
       dpi = 400, 
       device = "png")



