rm(list=ls())
library(data.table)
library(ggplot2)
library(dplyr)

part_1 = "/Users/hn/Desktop/Desktop/Kirti/check_point/my_aeolus_2015/all_local"
part_2 = "/pest_control/400_2nd_try_data/"
data_dir = paste0(part_1, part_2)
start_end_name = "start_end_"
full_window_name = "all_14_days_window_"
models = c("45", "85")

##############################################################
###############################
############################### GDD Cloud plots!
###############################
##############################################################
the_theme = theme(# panel.grid.major = element_line(size = 0.7),
                  # panel.grid.major = element_blank(),
                  plot.margin = unit(c(t=0.1, r=0.5, b=.1, l=0.2), "cm"),
                  panel.grid.minor = element_blank(),
                  panel.spacing=unit(.5, "cm"),
                  legend.title = element_text(face = "plain", size = 12),
                  legend.text = element_text(size = 10),
                  legend.position = "bottom",
                  strip.text = element_text(size = 12, face = "plain"),
                  axis.text = element_text(face = "plain", size = 10, color="black"),
                  axis.title.x = element_text(face = "plain", size=16, margin=margin(t=10, r=0, b=0, l=0)),
                  axis.title.y = element_text(face = "plain", size=16, margin=margin(t=0, r=10, b=0, l=0))
                    )
y_range = seq(0, 600, 50)
for (model in models){
  curr_data = data.table(readRDS(paste0(data_dir, full_window_name, model, ".rds")))
  curr_data = within(curr_data, remove(ID, PercLarvaGen1, dayofyear))
  curr_data <- curr_data[, .(CumDD = median(CumDDinF)), by = c("CountyGroup", 
                                                               "location", 
                                                               "ClimateScenario", 
                                                               "ClimateGroup", 
                                                               "year")]
  plot = ggplot(curr_data, aes(x=year, y=CumDD, fill=factor(ClimateGroup))) +
         labs(x = "Year", y = "Cumulative degree days (in F)", fill = "Climate group") +
         guides(fill=guide_legend(title="")) + 
         facet_grid(. ~ CountyGroup, scales="free") +
         stat_summary(geom="ribbon", fun.y=function(z) { quantile(z,0.5) }, 
                      fun.ymin=function(z) { quantile(z,0.1) }, 
                      fun.ymax=function(z) { quantile(z,0.9) }, alpha=0.4) +
         stat_summary(geom="ribbon", fun.y=function(z) { quantile(z,0.5) }, 
                      fun.ymin=function(z) { quantile(z,0.25) }, 
                      fun.ymax=function(z) { quantile(z,0.75) }, alpha=0.8) + 
         stat_summary(geom="line", fun.y=function(z) { quantile(z,0.5) })+ 
         scale_color_manual(values=c(rgb(29, 67, 111, max=255), 
                                     rgb(92, 160, 201, max=255), 
                                     rgb(211, 91, 76, max=255), 
                                     rgb(125, 7, 37, max=255)))+
         scale_fill_manual(values=c(rgb(29, 67, 111, max=255), 
                                    rgb(92, 160, 201, max=255), 
                                    rgb(211, 91, 76, max=255), 
                                    rgb(125, 7, 37, max=255)))+
         scale_x_continuous(breaks=seq(1980, 2100, 20)) +
         scale_y_continuous(breaks=y_range) +
         theme_bw() + the_theme
  out_name = paste0("gdd_cloud_", model, ".png" )
  part_1 <- "/Users/hn/Documents/GitHub/Kirti/codling_moth/to_write_paper/figures/"
  output_dir = paste0(part_1, "pest_window/3rd_try_plots/")
  ggsave(out_name, plot, path=output_dir, width=14, height=7, unit="in", dpi=500)
}
########################
######################## Smaller plot, in terms of saving size
########################
the_theme = theme(# panel.grid.major = element_line(size = 0.7),
                  # panel.grid.major = element_blank(),
                  plot.margin = unit(c(t=0.1, r=0.5, b=.1, l=0.2), "cm"),
                  panel.grid.minor = element_blank(),
                  panel.spacing=unit(.25, "cm"),
                  legend.title = element_text(face = "plain", size = 3),
                  legend.text = element_text(size = 5),
                  legend.key.size = unit(.3, "cm"),
                  legend.position = "bottom",
                  strip.text = element_text(size = 6, face = "plain"),
                  axis.ticks = element_line(color = "black", size = .2),
                  axis.text = element_text(face = "plain", size = 5, color="black"),
                  axis.title.x = element_text(face = "plain", size=8, margin=margin(t=5, r=0, b=-8, l=0)),
                  axis.title.y = element_text(face = "plain", size=8, margin=margin(t=0, r=5, b=0, l=0))
                    )
y_range = seq(0, 600, 50)

for (model in models){
  curr_data = data.table(readRDS(paste0(data_dir, full_window_name, model, ".rds")))
  curr_data = within(curr_data, remove(ID, PercLarvaGen1, dayofyear))
  curr_data <- curr_data[, .(CumDD = median(CumDDinF)), by = c("CountyGroup", 
                                                               "location", 
                                                               "ClimateScenario", 
                                                               "ClimateGroup", 
                                                               "year")]

  plot = ggplot(curr_data, aes(x=year, y=CumDD, fill=factor(ClimateGroup))) +
         labs(x = "Year", y = "Cumulative degree days (in F)", fill = "Climate group") +
         guides(fill=guide_legend(title="")) + 
         facet_grid(. ~ CountyGroup, scales="free") +
         stat_summary(geom="ribbon", fun.y=function(z) { quantile(z,0.5) }, 
                      fun.ymin=function(z) { quantile(z,0.1) }, 
                      fun.ymax=function(z) { quantile(z,0.9) }, alpha=0.4) +
         
         stat_summary(geom="ribbon", fun.y=function(z) { quantile(z,0.5) }, 
                      fun.ymin=function(z) { quantile(z,0.25) }, 
                      fun.ymax=function(z) { quantile(z,0.75) }, alpha=0.8) + 
         stat_summary(geom="line", size=.2, fun.y=function(z) { quantile(z,0.5) })+ 
         scale_color_manual(values=c(rgb(29, 67, 111, max=255), 
                                     rgb(92, 160, 201, max=255), 
                                     rgb(211, 91, 76, max=255), 
                                     rgb(125, 7, 37, max=255)))+
         scale_fill_manual(values=c(rgb(29, 67, 111, max=255), 
                                    rgb(92, 160, 201, max=255), 
                                    rgb(211, 91, 76, max=255), 
                                    rgb(125, 7, 37, max=255)))+
         scale_x_continuous(breaks=seq(1980, 2100, 20)) +
         scale_y_continuous(breaks=y_range) +
         theme_bw() + the_theme
  out_name = paste0("gdd_cloud_", model, "_1.png" )
  part_1 <- "/Users/hn/Documents/GitHub/Kirti/codling_moth/to_write_paper/figures/"
  output_dir = paste0(part_1, "pest_window/3rd_try_plots/")
  ggsave(out_name, plot, path=output_dir, width=7, height=3.5, unit="in", dpi=500)
}
