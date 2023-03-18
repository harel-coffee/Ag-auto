rm(list=ls())
library(data.table)
library(dplyr)
library(ggmap)
library(ggplot2)
options(digit=9)
options(digits=9)
#######################################################################
##                                                                   ##
##                    Function Definitions                           ##
##                                                                   ##
#######################################################################
historical_map <- function(data, color_col, min, max){
  data <- data %>% filter(year <= 2005)
  states <- map_data("state")
  states_cluster <- subset(states, region %in% c("oregon", "washington", "idaho"))

  if (color_col=="mean_over_model"){
    low_lim = min(data$mean_over_model)
    up_lim = max(data$mean_over_model)
    } else if (color_col=="mediam_over_model"){
      low_lim = min(data$median_over_model)
      up_lim = max(data$median_over_model)
  }
    
  data %>% ggplot() +
           geom_polygon(data = states_cluster, aes(x=long, y=lat, group = group),
                        fill = "grey", color = "black") +
           # aes_string to allow naming of column in function 
           geom_point(aes_string(x = "long", y = "lat",
                                 color = color_col), alpha = 0.4, size=.4) +
           coord_fixed(xlim = c(-124.5, -111.4),  ylim = c(41, 50.5), ratio = 1.3) +
           facet_grid(~ scenario ~ time_period) +
           ggtitle("Ensemble means") + 
           theme_bw() + 
           theme(legend.position = "bottom",
                 legend.title = element_blank(),
                 legend.key.size = unit(1.4, "line"),
                 plot.margin = margin(t=0, r=0.2, b=0, l=0.2, unit = 'cm')
                 # axis.text.x = element_text(size=3, face="plain", color="black"),
                 # axis.text.y = element_text(size=3, face="plain", color="black"),
                 # legend.margin = margin(t=0, r=0, b=-0.1, l=0, unit = 'cm')
                 ) +
           scale_color_gradient2(midpoint=(low_lim + up_lim)/2, low="red", mid="white", high="blue", 
                                 space ="Lab")
}

#######################################################################
##                                                                   ##
##                             Driver                                ##
##                                                                   ##
#######################################################################

time_types = c("non_overlapping") # , "overlapping"
model_types = c("dynamic_model_stats") # , "utah_model_stats"
main_in = "/Users/hn/Desktop/Desktop/Kirti/check_point/chilling"
file_name = "summary_comp.rds"

time_type = time_types[1]
model_type = model_types[1]

for (time_type in time_types){
  for (model_type in model_types){
    in_dir = file.path(main_in, time_type, model_type, file_name)
    out_dir = file.path(main_in, time_type, model_type, "/")
    
    datas = data.table(readRDS(in_dir))
    datas <- datas %>% filter(model != "observed")
    
    information = produce_data_4_plots(datas)

    safe_jan <- safe_box_plot(information[[1]])
    safe_feb <- safe_box_plot(information[[4]])
    
    output_name = paste0(time_type, "_", unlist(strsplit(model_type, "_"))[1], "_Jan.png")
    ggsave(output_name, safe_jan, path=out_dir, width=4, height=4, unit="in", dpi=400)

    output_name = paste0(time_type, "_", unlist(strsplit(model_type, "_"))[1], "_Feb.png")
    ggsave(output_name, safe_feb, path=out_dir, width=4, height=4, unit="in", dpi=400)
    
    # means over models
    mean_map_jan = ensemble_map(data=information[[2]], color_col="mean_over_model")
    mean_map_feb = ensemble_map(data=information[[5]], color_col="mean_over_model")

    output_name = paste0(time_type, "_", unlist(strsplit(model_type, "_"))[1], "_map_jan.png") 
    ggsave(output_name, mean_map_jan, path=out_dir, width=7, height=5.5, unit="in", dpi=400)

    output_name = paste0(time_type, "_", unlist(strsplit(model_type, "_"))[1], "_map_feb.png") 
    ggsave(output_name, mean_map_feb, path=out_dir, width=7, height=5.5, unit="in", dpi=400)
    print (out_dir)
    # map_plot(information[[3]]) # medians over models
  }
}

