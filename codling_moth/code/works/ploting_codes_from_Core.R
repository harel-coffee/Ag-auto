################################################################################
## *****************               Plotting Codes          *********************
################################################################################


#####################
####### Bloom Plots
#####################
plot_bloom1 <- function(data_dir, file_name = "vertdd_combined_CMPOP_", version, plot_path, output_name){
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
    stat_summary(geom="line", fun.y=function(z) { quantile(z,0.5) }, size = 1)+ #, aes(color=factor(Timeframe))) + , # aes(color=factor(ClimateGroup))
    scale_color_manual(values=c(rgb(29, 67, 111, max=255), rgb(92, 160, 201, max=255), rgb(211, 91, 76, max=255), rgb(125, 7, 37, max=255)))+#c("black", "red","darkgreen","blue")) +
    scale_fill_manual(values=c(rgb(29, 67, 111, max=255), rgb(92, 160, 201, max=255), rgb(211, 91, 76, max=255), rgb(125, 7, 37, max=255)))+#c("black", "red","darkgreen","blue")) +
    facet_grid(. ~ variable ~ CountyGroup, scales = "free") +
    #xlim(45, 165) +
    scale_x_continuous(breaks=seq(45, 150, 10), limits = c(45, 150)) +
    theme_bw() +
    labs(x = "Julian Day", y = "Proportion Full Bloom Completed", fill = "Climate Group") +
    theme(
      panel.grid.major = element_line(size = 0.7),
      axis.text = element_text(face = "bold", size = 18),
      axis.title = element_text(face = "bold", size = 20),
      legend.title = element_text(face = "bold", size = 20),
      legend.text = element_text(size = 20),
      legend.position = "top",
      #plot.title = element_text(face = "bold", size = 18, hjust = 0.5),
      strip.text = element_text(size = 18, face = "bold"))

  ggsave(output_name, p1, path=plot_path, width = 45, height = 22, units = "cm")
}


plot_bloom2 <- function(data_dir, file_name = "vertdd_combined_CMPOP_", version, plot_path, output_name){
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
    scale_x_continuous(breaks=seq(45, 150, 10), limits = c(45, 150)) +
    theme_bw() +
    labs(x = "Julian Day", y = "Proportion Completing Full Bloom", color = "Climate Group") +
    theme(
      panel.grid.major = element_line(size = 0.7),
      axis.text = element_text(face = "bold", size = 18),
      axis.title = element_text(face = "bold", size = 20),
      legend.title = element_text(face = "bold", size = 20),
      legend.text = element_text(size = 20),
      legend.position = "top",
      #plot.title = element_text(face = "bold", size = 18, hjust = 0.5),
      strip.text = element_text(size = 18, face = "bold"))

  ggsave(output_name, p1, path=plot_path, width = 45, height = 22, units = "cm")
}



#####################
#######
#####################

plot_cumdd_eggHatch <- function(input_dir, file_name ="combined_CMPOP_", version, output_dir, output_type="cumdd"){
  if (output_type == "eggHatch"){
    data = compute_cumdd(input_dir=data_dir, file_name="combined_CMPOP_", version)
    data$CountyGroup = as.character(data$CountyGroup)
    data[CountyGroup == 1]$CountyGroup = 'Cooler Areas'
    data[CountyGroup == 2]$CountyGroup = 'Warmer Areas'
    data = melt(data, id = c("ClimateGroup", "CountyGroup", "latitude", "longitude", "ClimateScenario", "year", "dayofyear"))
    plot = ggplot(data[value >=0.01 & value <.98 & dayofyear <300], aes(x=dayofyear, y=value, fill=factor(variable))) +
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
                 fun.y=function(z) { quantile(z,0.5) })+ #, aes(color=factor(Timeframe))) + 
    
    scale_color_manual(values=c(rgb(29, 67, 111, max=255), 
                       rgb(92, 160, 201, max=255), 
                       rgb(211, 91, 76, max=255), 
                       rgb(125, 7, 37, max=255))) + #c("black", "red","darkgreen","blue")) +
    scale_fill_manual(values=c(rgb(29, 67, 111, max=255), 
                               rgb(92, 160, 201, max=255), 
                               rgb(211, 91, 76, max=255), 
                               rgb(125, 7, 37, max=255)))+#c("black", "red","darkgreen","blue")) +
    
    facet_grid(. ~ ClimateGroup ~ CountyGroup, scales="free") +
    scale_x_continuous(breaks=seq(0,300,50)) +
    theme_bw() +
    geom_vline(xintercept=c(100,150,200,250,300), linetype="solid", color ="grey")+
    geom_vline(xintercept=c(120,226), linetype="solid", color ="red")+
    geom_vline(xintercept=seq(70,300,10), linetype="dotdash")+
    geom_hline(yintercept=c(.25,.5,.75), linetype="dotted", color = "black")+
    labs(x = "Julian Day", y = "Cumulative population fraction", fill = "Larva Generation") +
    theme(
      panel.grid.major = element_line(size = 0.7),
      axis.text = element_text(face = "bold", size = 18),
      axis.title = element_text(face = "bold", size = 20),
      legend.title = element_text(face = "bold", size = 20),
      legend.text = element_text(size = 20),
      legend.position = "top",
      #plot.title = element_text(face = "bold", size = 18, hjust = 0.5),
      strip.text = element_text(size = 18, face = "bold"))
  out_name = paste0("plot_", output_type, "_", version ,".png")
  ggsave(out_name, plot, path=output_dir, width = 45, height = 22, units = "cm")
  #saveRDS(plot, "cumdd_plot.rds")
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
    data = subset(data, select = c("ClimateGroup", "CountyGroup", "latitude", 
                                   "longitude", "ClimateScenario", "year", "dayofyear", "CumDDinF"))
    data$CumDD = data$CumDDinF

    data <- data[, .(CumDD = median(CumDDinF)), by = c("CountyGroup", "latitude", "longitude", "ClimateScenario", "ClimateGroup", "dayofyear")]
    # data <- data[, .(CumDD =   mean(CumDDinF)), by = c("CountyGroup", "latitude", "longitude", "ClimateScenario", "ClimateGroup", "dayofyear")]

    if (version == "rcp85"){
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
        
        facet_grid(. ~ CountyGroup, scales="free") +
        scale_x_continuous(breaks=seq(0, 370, 25)) +
        scale_y_continuous(breaks=seq(0, 5750, 250)) +
        #scale_y_continuous(breaks=seq(0,4500,250)) +
        theme_bw() +
        #geom_vline(xintercept=c(100,150,200,250,300), linetype="solid", color ="grey")+
        #geom_vline(xintercept=c(120,226), linetype="solid", color ="red")+
        #geom_vline(xintercept=seq(70,300,10), linetype="dotdash")+
        #geom_hline(yintercept=c(.25,.5,.75), linetype="dotted", color = "black")+
        labs(x = "Julian Day", y = "Cumulative Degree Days (in F)", fill = "Climate Group") +
        theme(
          panel.grid.major = element_line(size = 0.7),
          axis.text = element_text(face = "bold", size = 18),
          axis.title = element_text(face = "bold", size = 20),
          legend.title = element_text(face = "bold", size = 20),
          legend.text = element_text(size = 20),
          legend.position = "top",
          #plot.title = element_text(face = "bold", size = 18, hjust = 0.5),
          strip.text = element_text(size = 18, face = "bold"))
        } else if (version == "rcp45"){
          plot = ggplot(data, aes(x=dayofyear, y=CumDD, fill=factor(ClimateGroup))) +
          #geom_line(aes(fill=factor(Timeframe), color=factor(Timeframe) )) +
          stat_summary(geom="ribbon", fun.y=function(z) { quantile(z,0.5) }, fun.ymin=function(z) { quantile(z,0.1) }, fun.ymax=function(z) { quantile(z,0.9) }, alpha=0.4) +
          stat_summary(geom="ribbon", fun.y=function(z) { quantile(z,0.5) }, fun.ymin=function(z) { quantile(z,0.25) }, fun.ymax=function(z) { quantile(z,0.75) }, alpha=0.8) + 
          stat_summary(geom="line", fun.y=function(z) { quantile(z,0.5) })+ #, aes(color=factor(Timeframe))) + 
          
          scale_color_manual(values=c(rgb(29, 67, 111, max=255), rgb(92, 160, 201, max=255), rgb(211, 91, 76, max=255), rgb(125, 7, 37, max=255)))+#c("black", "red","darkgreen","blue")) +
          scale_fill_manual(values=c(rgb(29, 67, 111, max=255), rgb(92, 160, 201, max=255), rgb(211, 91, 76, max=255), rgb(125, 7, 37, max=255)))+#c("black", "red","darkgreen","blue")) +
          
          facet_grid(. ~ CountyGroup, scales="free") +
          scale_x_continuous(breaks=seq(0,370,25)) +
          #scale_y_continuous(breaks=seq(0,5750,250)) +
          scale_y_continuous(breaks=seq(0,4500,250)) +
          theme_bw() +
          #geom_vline(xintercept=c(100,150,200,250,300), linetype="solid", color ="grey")+
          #geom_vline(xintercept=c(120,226), linetype="solid", color ="red")+
          #geom_vline(xintercept=seq(70,300,10), linetype="dotdash")+
          #geom_hline(yintercept=c(.25,.5,.75), linetype="dotted", color = "black")+
          labs(x = "Julian Day", y = "Cumulative Degree Days (in F)", fill = "Climate Group") +
          theme(
            panel.grid.major = element_line(size = 0.7),
            axis.text = element_text(face = "bold", size = 18),
            axis.title = element_text(face = "bold", size = 20),
            legend.title = element_text(face = "bold", size = 20),
            legend.text = element_text(size = 20),
            legend.position = "top",
            #plot.title = element_text(face = "bold", size = 18, hjust = 0.5),
            strip.text = element_text(size = 18, face = "bold"))
        }
    out_name = paste0("plot_", output_type, "_", version, ".png")
    ggsave(out_name, plot, path=output_dir, width = 45, height = 22, units = "cm")

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
    #geom_vline(xintercept=c(100,150,200,250,300), linetype="solid", color ="grey")+
    #geom_vline(xintercept=c(120,226), linetype="solid", color ="red")+
    #geom_vline(xintercept=seq(70,300,10), linetype="dotdash")+
    #geom_hline(yintercept=c(.25,.5,.75), linetype="dotted", color = "black")+
    labs(x = "Julian Day", y = "Cumulative Degree Days (in F)", fill = "Climate Group") +
    theme(
      panel.grid.major = element_line(size = 0.7),
      axis.text = element_text(face = "bold", size = 18),
      axis.title = element_text(face = "bold", size = 20),
      legend.title = element_text(face = "bold", size = 20),
      legend.text = element_text(size = 20),
      legend.position = "top",
      #plot.title = element_text(face = "bold", size = 18, hjust = 0.5),
      strip.text = element_text(size = 18, face = "bold"))
  out_name = paste0("plot_cumdd_" , version, "_type", output_type ,".png")
  ggsave(out_name, plot, path=output_dir, width = 45, height = 22, units = "cm")
  }
}
