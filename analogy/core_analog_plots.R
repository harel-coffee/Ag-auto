library(data.table)
library(dplyr)
library(ggmap)
library(ggpubr)

options(digits=9)
options(digit=9)
#################################################################################
#################
#################    Donut functions
#################
#################################################################################
plot_f_h_2_features_all_models <- function(future_dt, top3_data, hist_dt){
  # input: future_dt: future features for one fip (target county), one time period, all models
  #        top3_data: data table including top three similar analogs of given county.
  #        hist_dt: all historical data
  # output: scatter plot of both of these in the same plot
  all_models <- sort(unique(future_dt$model))

  for (a_model in all_models){

    ## pick up a model, and its corresponding analog
    # future
    future_dt_a_model <- future_dt %>% filter(model == a_model) %>% data.table()
    
    # History
    most_similar_county_fip <- top3_data$top_1_fip[top3_data$model == a_model]
    hist_dt_curr <- hist_dt %>% filter(fips == most_similar_county_fip) %>% data.table()
    
    assign(x= paste0("plot_", gsub("-", "_", a_model)),
           value = {plot_f_h_2_features_1_model(future_dt_a_model, hist_dt_curr)})
  }
  assign(x = "plot" , 
         value={ggarrange(plotlist = list(plot_bcc_csm1_1_m, 
                                          plot_BNU_ESM, 
                                          plot_CanESM2,
                                          plot_CNRM_CM5, 
                                          plot_GFDL_ESM2G, 
                                          plot_GFDL_ESM2M
                                          ),
                          ncol = 1, nrow = length(all_models), 
                          common.legend = TRUE, 
                          legend = "bottom")})
  return(plot)
}

plot_f_h_2_features_1_model <- function(f_data, hist_data){
  # input: future_dt: future features for one fip (target county), one time period, one model
  #        top3_data: data table including top three similar analogs of given county.
  #        hist_dt: all historical data
  # output: scatter plot of both of these in the same plot
  
  # extract model name for use in title
  model <- unique(f_data$model)
  time_frame <- unique(f_data$time_period)
  target_county_name <- unique(f_data$st_county)
  analog_county_name <- unique(hist_data$st_county)

  target_county_name <- paste0(unlist(strsplit(target_county_name, "_"))[2], ", ",
                               unlist(strsplit(target_county_name, "_"))[1])
  analog_county_name <- paste0(unlist(strsplit(analog_county_name, "_"))[2], ", ",
                               unlist(strsplit(analog_county_name, "_"))[1])

  mini_inf <- paste0(" (", time_frame, ", ", model, ")")
  plt_title <- paste0(target_county_name, mini_inf)
  plt_subtitle <- paste0("Analog: ", analog_county_name, " (1979-2015, observed)")
  plot_data <- rbind(hist_data, f_data)

  the_theme <- theme(plot.title = element_text(size = 30, face="bold", color="black"),
                     plot.subtitle = element_text(size = 26, face="plain", color="black"),
                     axis.text.x = element_text(size = 20, face = "bold", color="black"),
                     axis.text.y = element_text(size = 20, face = "bold", color="black"),
                     axis.title.x = element_text(size = 30, face = "bold", color="black", 
                                                 margin = margin(t=8, r=0, b=8, l=0)),
                     axis.title.y = element_text(size = 30, face = "bold", color="black",
                                                 margin = margin(t=0, r=8, b=0, l=0)),
                     strip.text = element_text(size=30, face = "bold"),
                     legend.margin=margin(t=.1, r=0, b=0, l=0, unit='cm'),
                     legend.title = element_blank(),
                     legend.position="bottom", 
                     legend.key.size = unit(1.5, "line"),
                     legend.spacing.x = unit(.05, 'cm'),
                     legend.text=element_text(size=12),
                     panel.spacing.x =unit(.75, "pt"))

  plt <- ggplot(data = plot_data) +
         geom_point(aes(x = CumDDinF_Aug23, y = yearly_precip, 
                        fill = time_period),
                    alpha = .5, shape = 21, size=9) +
         ylab("annual precip. (mm)") +
         xlab("pest pressure") + # Cum. DD (F) by Aug 23
         ggtitle(label = plt_title,
                 subtitle = plt_subtitle) + 
         guides(colour = guide_legend(override.aes = list(size=100))) + 
         the_theme
  return(plt)
}

plot_fng_dens <- function(data_dt){
  color_ord = c("red", "dodgerblue") #, "olivedrab4", grey47

  if("CumDDinF_Aug23" %in% colnames(data_dt)){
     x_variable_1 <- "CumDDinF_Aug23"
     x_variable_2 <- "yearly_precip"
     } else {
        x_variable_1 <- "mean_CumDDinF_Aug23"
        x_variable_2 <- "mean_yearly_precip"
  }
  x_lab_1 <- "pest pressure"
  x_lab_2 <- "annual precip. (mm)"
  y_lab <- "density"
 
  simils <- unlist(seperate_1D_similarities(data_dt))
  press_simil <- signif(simils[1] * 100, 4)
  precip_simil <- signif(simils[2] * 100, 4)
  press_ann <- paste0(press_simil, "% similarity")
  precip_ann <- paste0(precip_simil, "% similarity")
  
  pp_label_x <- find_x_position(data_dt, x_variable_1)
  prec_label_x <- find_x_position(data_dt, x_variable_2)

  pp_y_pos <- find_y_position(data_dt, x_variable_1)
  prec_y_pos <- find_y_position(data_dt, x_variable_2)

  the_th <- theme(plot.margin = unit(c(t=.2, r=.2, b=.2, l=0.2), "cm"),
                  panel.border = element_rect(fill=NA, size=.3),
                  panel.grid.major = element_line(size = 0.05),
                  panel.grid.minor = element_blank(),
                  panel.spacing = unit(.35, "line"),
                  legend.position = "bottom", 
                  legend.key.size = unit(.7, "line"),
                  legend.spacing.x = unit(.1, 'line'),
                  panel.spacing.y = unit(.5, 'line'),
                  legend.text = element_text(size = 7, face="bold"),
                  legend.margin = margin(t=.2, r=0, b=.2, l=4, unit = 'line'),
                  legend.title = element_blank(),
                  plot.title = element_text(size = 7, face = "bold"),
                  plot.subtitle = element_text(face = "bold"),
                  strip.text.x = element_text(size=7, face = "bold"),
                  strip.text.y = element_text(size=7, face = "bold"),
                  axis.ticks = element_line(size = .1, color = "black"),
                  axis.text.y = element_blank(), # element_text(size = 8, face = "bold", color = "black"),
                  axis.text.x = element_blank(), # element_text(size = 8, face = "bold", color="black"),
                  axis.title.y = element_blank(), # element_text(size = 10, face = "bold", margin = margin(t=0, r=4, b=0, l=0)),
                  axis.title.x = element_text(size = 12, face = "bold", margin = margin(t=4, r=0, b=4, l=0)),
                  )
  pp <- ggplot() +
        geom_density(data = data_dt, 
                     aes(x=get(x_variable_1), fill=model, color=model), 
                     alpha = 0.7) +
        scale_color_manual(values=color_ord) +
        labs(x=x_lab_1, y=y_lab) +
        geom_label(aes(x=pp_label_x , y=pp_y_pos, label = press_ann),
                   size = 5, fill="white",
                   label.padding = unit(0.55, "lines")) +
        the_th
  
  pre <- ggplot()+ 
         geom_density(data = data_dt, 
                      aes(x=get(x_variable_2), fill=model, color=model), 
                          alpha = 0.7) + 
         scale_color_manual(values=color_ord) +
         labs(x=x_lab_2, y=y_lab) +
         geom_label(aes(x=prec_label_x , y=prec_y_pos, label = precip_ann),
                   size = 5, fill="white",
                   label.padding = unit(0.55, "lines")) +
        the_th

  pp_pre <- ggarrange(pp, pre, ncol=1, nrow=2, 
                      common.legend=TRUE,
                      legend="none")
  return(pp_pre)
}

plot_the_margins_cowplot <- function(data_dt, contour_plot){
  color_ord = c("red", "dodgerblue") #, "olivedrab4", grey47

  if("CumDDinF_Aug23" %in% colnames(data_dt)){
     x_variable_1 <- "CumDDinF_Aug23"
     x_variable_2 <- "yearly_precip"
     } else {
        x_variable_1 <- "mean_CumDDinF_Aug23"
        x_variable_2 <- "mean_yearly_precip"
  }
  DD_plt <- cowplot::axis_canvas(contour_plot, axis="x") + 
            geom_density(data = data_dt, 
                         aes(x=get(x_variable_1), fill=model, color=model), 
                         alpha = 0.7) +
            scale_color_manual(values=color_ord)
  
  preip_plt <- cowplot::axis_canvas(contour_plot, axis="y", coord_flip=TRUE) + 
               geom_density(data = data_dt, 
                            aes(x=get(x_variable_2), fill=model, color=model), 
                            alpha = 0.7) + 
               coord_flip() +
               scale_color_manual(values=color_ord)

  p1 <- cowplot::insert_xaxis_grob(contour_plot, DD_plt, grid::unit(.2, "null"), position= "top")
  p2 <- cowplot::insert_yaxis_grob(p1, preip_plt, grid::unit(.2, "null"), position= "right")

  # return (contour_with_margins)
  return(p2)
}

plot_the_margins <- function(data_dt, contour_plot){
  color_ord = c("red", "dodgerblue") #, "olivedrab4", grey47

  if ("CumDDinF_Aug23" %in% colnames(data_dt)){
    x_variable_1 <- "CumDDinF_Aug23"
    x_variable_2 <- "yearly_precip"
    
   } else {
    x_variable_1 <- "mean_CumDDinF_Aug23"
    x_variable_2 <- "mean_yearly_precip"
  }
  empty <- ggplot()+ 
           geom_point(aes(1,1), colour="white")+
           theme(axis.ticks=element_blank(), 
                 panel.background=element_blank(), 
                 axis.text.x=element_blank(), axis.text.y=element_blank(),
                 axis.title.x=element_blank(), axis.title.y=element_blank())

  DD_plt <- ggplot(data_dt, aes(x = get(x_variable_1), fill=model, color=model)) +
            geom_density(alpha = 0.5) + 
            scale_color_manual(values=color_ord) + 
            guides(colour = guide_legend(reverse = TRUE), fill=guide_legend(reverse = TRUE)) + 
            theme(plot.margin = unit(c(t=0.2, r=1, b=0.5, l=.5), "pt"),
                  legend.position = "none",
                  axis.ticks.x = element_blank(),
                  axis.ticks.y = element_blank(),
                  axis.text.x = element_text(size=15, face="plain", color="black"),
                  axis.text.y = element_text(size=15, face="plain", color="black"),
                  axis.title.x = element_blank(),
                  axis.title.y = element_blank()
                  )
  
  preip_plt <- ggplot(data_dt, aes(x = get(x_variable_2), fill=model, color=model)) +
               geom_density(alpha = 0.5) + 
               scale_color_manual(values=color_ord) + 
               guides(colour = guide_legend(reverse = TRUE), fill=guide_legend(reverse = TRUE)) + 
               coord_flip() + 
               theme(plot.title = element_text(size=20, face="bold"),
                     plot.margin = unit(c(t=.1, r=.5, b=1.2, l=0.2), "pt"),
                     legend.position = "none",
                     axis.ticks.x = element_blank(),
                     axis.ticks.y = element_blank(),
                     axis.text.x = element_text(size=15, face="plain", color="black"),
                     axis.text.y = element_text(size=15, face="plain", color="black"),
                     axis.title.x = element_blank(),
                     axis.title.y = element_blank()
                     )  

  contour_with_margins <- ggarrange(DD_plt, 
                                    empty, 
                                    contour_plot,
                                    preip_plt, # This is on right
                                    ncol=2, nrow=2, 
                                    widths = c(4, 1), 
                                    heights = c(1, 4))

  return (contour_with_margins)
}

plot_the_contour_one_filling <- function(data_dt, con_title, con_subT){
  # , v_line_quantiles=c(0.1, 0.9)
  color_ord = c("red", "dodgerblue")
  the_theme <- theme(# plot.title = element_text(size=20, face="bold"),
                     plot.margin = unit(c(t=.1, r=1, b=0, l=.5), "pt"),
                     legend.spacing.x = unit(0.4, 'cm'),
                     legend.title = element_blank(),
                     legend.position = "bottom",
                     legend.key.size = unit(1, "line"),
                     legend.text = element_text(size=15, face="plain"),
                     legend.margin = margin(t=.1, r=0, b=.1, l=0, unit = 'cm'),
                     axis.ticks.x = element_blank(),
                     axis.ticks.y = element_blank(),
                     axis.text.x = element_text(size=12, face="plain", color="black"),
                     axis.text.y = element_text(size=12, face="plain", color="black"),
                     axis.title.x = element_text(size=18, face="plain", color="black",
                                                 margin = margin(t=15, r=0, b=0, l=0)),
                     axis.title.y = element_text(size=18, face="plain", color="black", 
                                                 margin = margin(t=0, r=15, b=0, l=0)))

  y_lab <- "annual precip. (mm)"
  x_lab <- "pest pressure"
   
  if ("CumDDinF_Aug23" %in% colnames(data_dt)){
    x_variable <- "CumDDinF_Aug23"
    y_variable <- "yearly_precip"
   } else {
    x_variable <- "mean_CumDDinF_Aug23"
    y_variable <- "mean_yearly_precip"
  }  
  contour_plt <- ggplot(data_dt, aes(x = get(x_variable), y = get(y_variable))) + 
                 ylab(y_lab) + xlab(x_lab) + 
                 stat_density_2d(aes(fill = stat(level), colour = model), 
                                 alpha = .4, contour = TRUE, geom = "polygon") + 
                 scale_fill_viridis_c(guide = FALSE, aesthetics = "fill") + 
                 scale_color_manual(values = color_ord) + 
                 guides(color = guide_legend(reverse = TRUE)) +
                 the_theme

  return(contour_plt)
}

##################################################################
map_of_all_models_anlgs_freq_color <- function(a_dt, county2, title_p, target_county_map_info){
  title_p <- unlist(strsplit(title_p, " "))
  title_p <- title_p[-4]
  title_p <- paste(title_p, collapse=" ")

  # count_of_county_rep <- a_dt %>% group_by(model) %>% count(analog_NNs_county)

  color_ord = c("grey47", "dodgerblue", "olivedrab4", "yellow", "orange2", "blue4") # 
  county_fill_in = c("cyan3", "dodgerblue", "olivedrab4", "yellow", "orange2", "blue4") # 
  
  categ_lab = c( "bcc-csm1-1-m", "BNU-ESM", "CanESM2", "CNRM-CM5", "GFDL-ESM2G", "GFDL-ESM2M")
  
  cols_4_arrow <- c("long", "lat", "group", "order", "region", "subregion", "polyname")
  
  start_end_df <- target_county_map_info[1,]
  start_end_df <- start_end_df[colnames(start_end_df) %in% cols_4_arrow]

  for (modelsss in categ_lab){
    aaa <- a_dt %>% filter(model == modelsss)
    # aaa <- aaa[1,]
    avg_long <- mean(aaa$long); avg_lat <- mean(aaa$lat)
    aaa$long <- avg_long; aaa$lat <- avg_lat
    aaa <- aaa[colnames(aaa) %in% cols_4_arrow]
    start_end_df <- rbind(start_end_df, aaa[1,])
  }
  #***************************************
  #      replce centroids of each county!
  #***************************************
  start_end_df <- find_target_centroids(start_end_df)
  unique_start_end <- unique(start_end_df)

  # Fix the names of subregions, first letter capitalization
  for (ii in 1:nrow(unique_start_end)){
    c <- unique_start_end[ii, "subregion"]
    c <- unlist(strsplit(c, " "))
    for (jj in 1:length(c)){
     c[jj] <- Hmisc::capitalize(c[jj])
     }
    unique_start_end[ii, "subregion"] <- paste(c, collapse = " ")
  }

  # start and end of curve cannot be the same! 
  # make the starting point different in this case:
  if (unique(a_dt$query_county) %in% a_dt$analog_NNs_county){
    start_end_df$long[1] <- start_end_df$long[1] + .1
    start_end_df$lat[1] <- start_end_df$lat[1] + .1
  }

  count_of_counties <- a_dt %>% 
                       group_by(model) %>% 
                       count(analog_NNs_county) %>% 
                       group_by(analog_NNs_county) %>% 
                       count() %>% data.table()
  a_dt$county_count = 0L

  for (ii in seq(1:nrow(count_of_counties))){
    a_dt$county_count[a_dt$analog_NNs_county == as.integer(count_of_counties[ii, 1])] <- 
                                                  as.character(count_of_counties[ii, 2])
  }
  
  arrow_size = 0.02
  arrow_color <- "black"
  
  curr_plot <- ggplot(a_dt, aes(long, lat, group = group)) + 
               geom_polygon(data = county2, color="black", size=.05, fill="lightgrey") +
               geom_polygon(aes(fill = county_count), colour = rgb(1, 1, .11, .2), size = .01) +
               geom_polygon(data = target_county_map_info, color="red", size = .75, fill=NA) +
               borders("state") +
               ggrepel::geom_label_repel(data = unique_start_end,
                               aes(x = long, y = lat, label = subregion), 
                               fontface = "bold", box.padding = 0.2, # , 
                               label.r = 0.55, hjust=-.5, # vjust= -1,
                               # nudge_x = seq(1, .5 + nrow(unique_start_end)/2, by=0.5),
                               # nudge_y = seq(- (.5 + nrow(unique_start_end)/2 ), -1, by=0.5)
                               ) + 
               coord_quickmap() + 
               theme(plot.title = element_text(size=18, face="bold"),
                     plot.subtitle = element_text(size=14, face="bold"),
                     plot.margin = unit(c(t=2, r=10, b=-1, l=10), "pt"),
                     legend.spacing.x = unit(5, 'pt'),
                     # legend.spacing.y = unit(1, 'cm'),
                     legend.title = element_text(size=15, face="bold"),
                     legend.position = c(.87, .2), # 
                     legend.background = element_rect(fill = "grey92"),
                     legend.key.size = unit(2, "line"),
                     legend.text = element_text(size=15, face="plain"),
                     axis.text.x = element_blank(),
                     axis.text.y = element_blank(),
                     axis.ticks.x = element_blank(),
                     axis.ticks.y = element_blank(),
                     axis.title.x = element_blank(),
                     axis.title.y = element_blank()) + 
               ggtitle(title_p) + 
               # scale_fill_continuous(limits = c(1, 6), breaks = 1:6,
               #                        guide = guide_colourbar(nbin=6, draw.ulim = TRUE, 
               #                                                draw.llim = TRUE)) + 
               scale_fill_manual(values = county_fill_in, name = "freq. of analog", 
                                 guide = guide_legend(reverse = TRUE)) + 
               geom_curve(aes( x = start_end_df[1, "long"], y = start_end_df[1, "lat"], xend = start_end_df[2, "long"], yend = start_end_df[2, "lat"]), 
                          colour = arrow_color, data = start_end_df,
                          arrow = arrow(length = unit(arrow_size, "npc")), 
                          curvature = compute_curvature (x = start_end_df[1, "long"], 
                                                         y = start_end_df[1, "lat"], 
                                                         xend = start_end_df[2, "long"], 
                                                         yend = start_end_df[2, "lat"])
                          ) + 
               geom_curve(aes( x = start_end_df[1, "long"], y = start_end_df[1, "lat"], xend = start_end_df[3, "long"], yend = start_end_df[3, "lat"]), 
                          colour = arrow_color, data = start_end_df,
                          arrow = arrow(length = unit(arrow_size, "npc")), 
                          curvature = compute_curvature (x = start_end_df[1, "long"], 
                                                         y = start_end_df[1, "lat"], 
                                                         xend = start_end_df[3, "long"], 
                                                         yend = start_end_df[3, "lat"])
                          ) + 
               geom_curve(aes( x = start_end_df[1, "long"], y = start_end_df[1, "lat"], xend = start_end_df[4, "long"], yend = start_end_df[4, "lat"]), 
                           colour = arrow_color, data = start_end_df,
                           arrow = arrow(length = unit(arrow_size, "npc")), 
                           curvature = compute_curvature (x = start_end_df[1, "long"], 
                                                          y = start_end_df[1, "lat"], 
                                                          xend = start_end_df[4, "long"], 
                                                          yend = start_end_df[4, "lat"])
                          ) + 
               geom_curve(aes( x = start_end_df[1, "long"], y = start_end_df[1, "lat"], xend = start_end_df[5, "long"], yend = start_end_df[5, "lat"]), 
                          colour = arrow_color, data = start_end_df,
                          arrow = arrow(length = unit(arrow_size, "npc")), 
                          curvature = compute_curvature (x = start_end_df[1, "long"], 
                                                         y = start_end_df[1, "lat"], 
                                                         xend = start_end_df[5, "long"], 
                                                         yend = start_end_df[5, "lat"])
                          ) + 
               geom_curve(aes( x = start_end_df[1, "long"], y = start_end_df[1, "lat"], xend = start_end_df[6, "long"], yend = start_end_df[6, "lat"]), 
                          colour = arrow_color, data = start_end_df,
                          arrow = arrow(length = unit(arrow_size, "npc")), 
                          curvature = compute_curvature (x = start_end_df[1, "long"], 
                                                         y = start_end_df[1, "lat"], 
                                                         xend = start_end_df[6, "long"], 
                                                         yend = start_end_df[6, "lat"])
                          ) + 
               geom_curve(aes( x = start_end_df[1, "long"], y = start_end_df[1, "lat"], xend = start_end_df[7, "long"], yend = start_end_df[7, "lat"]), 
                           colour = arrow_color, data = start_end_df,
                           arrow = arrow(length = unit(arrow_size, "npc")), 
                           curvature = compute_curvature (x = start_end_df[1, "long"], 
                                                          y = start_end_df[1, "lat"], 
                                                          xend = start_end_df[7, "long"], 
                                                          yend = start_end_df[7, "lat"])
                         )

  curr_plot
  # rm(start_end_df)
  return(curr_plot) 
}

map_of_all_models_anlgs <- function(a_dt, county2, title_p, target_county_map_info){
  title_p <- unlist(strsplit(title_p, " "))
  title_p <- title_p[-4]
  title_p <- paste(title_p, collapse=" ")

  # color_ord = c("grey47" , "dodgerblue", "olivedrab4", "yellow", "orange2", "blue4") # 
  categ_lab = c( "bcc-csm1-1-m", "BNU-ESM", "CanESM2", "CNRM-CM5", "GFDL-ESM2G", "GFDL-ESM2M")
  
  cols_4_arrow <- c("long", "lat", "group", "order", "region", "subregion", "polyname")
  start_end_df <- target_county_map_info[1,]
  start_end_df <- start_end_df[colnames(start_end_df) %in% cols_4_arrow]

  for (modelsss in categ_lab){
    aaa <- a_dt %>% filter(model == modelsss)
    # avg_long <- mean(aaa$long); avg_lat <- mean(aaa$lat)
    aaa <- aaa[1,]
    # aaa$long <- avg_long; aaa$lat <- avg_lat
    aaa <- aaa[colnames(aaa) %in% cols_4_arrow]
    start_end_df <- rbind(start_end_df, aaa[1,])
  }
  
  arrow_size = 0.02
  
  curr_plot <- ggplot(a_dt, aes(long, lat, group = group)) + 
               geom_polygon(data = county2, color="black", size=.05, fill="lightgrey") +
               geom_polygon(aes(fill = model), colour = rgb(1, 1, .11, .2), size = .01) +
               geom_polygon(data = target_county_map_info, color="red", size = .75, fill=NA) +
               borders("state") +
               coord_quickmap() + 
               theme(plot.title = element_text(size=18, face="bold"),
                     plot.subtitle = element_text(size=14, face="bold"),
                     plot.margin = unit(c(t=2, r=4, b=-1, l=0), "pt"),
                     legend.spacing.x = unit(5, 'pt'),
                     # legend.spacing.y = unit(1, 'cm'),
                     legend.title = element_blank(),
                     legend.position = c(.87, .2), # 
                     legend.background = element_rect(fill = "grey92"),
                     legend.key.size = unit(1, "line"),
                     legend.text = element_text(size=15, face="plain"),
                     axis.text.x = element_blank(),
                     axis.text.y = element_blank(),
                     axis.ticks.x = element_blank(),
                     axis.ticks.y = element_blank(),
                     axis.title.x = element_blank(),
                     axis.title.y = element_blank()) + 
               scale_fill_manual(values = color_ord,
                                 name = "Model", 
                                 labels = categ_lab) +
               ggtitle(title_p) + 
               geom_curve(aes( x = start_end_df[1, "long"], y = start_end_df[1, "lat"], 
                               xend = start_end_df[2, "long"], yend = start_end_df[2, "lat"]), 
                          colour = "black", data = start_end_df,
                          arrow = arrow(length = unit(arrow_size, "npc")), 
                          curvature = compute_curvature (x = start_end_df[1, "long"], 
                                                         y = start_end_df[1, "lat"], 
                                                         xend = start_end_df[2, "long"], 
                                                         yend = start_end_df[2, "lat"])
                          ) + 
               geom_curve(aes( x = start_end_df[1, "long"], y = start_end_df[1, "lat"], 
                               xend = start_end_df[3, "long"], yend = start_end_df[3, "lat"]), 
                          colour = "black", data = start_end_df,
                          arrow = arrow(length = unit(arrow_size, "npc")), 
                          curvature = compute_curvature (x = start_end_df[1, "long"], 
                                                         y = start_end_df[1, "lat"], 
                                                         xend = start_end_df[3, "long"], 
                                                         yend = start_end_df[3, "lat"])
                          ) + 
               geom_curve(aes( x = start_end_df[1, "long"], y = start_end_df[1, "lat"], 
                               xend = start_end_df[4, "long"], yend = start_end_df[4, "lat"]), 
                          colour = "black", data = start_end_df,
                          arrow = arrow(length = unit(arrow_size, "npc")), 
                          curvature = compute_curvature (x = start_end_df[1, "long"], 
                                                         y = start_end_df[1, "lat"], 
                                                         xend = start_end_df[4, "long"], 
                                                         yend = start_end_df[4, "lat"])
                          ) + 
               geom_curve(aes( x = start_end_df[1, "long"], y = start_end_df[1, "lat"], 
                               xend = start_end_df[5, "long"], yend = start_end_df[5, "lat"]), 
                          colour = "black", data = start_end_df,
                          arrow = arrow(length = unit(arrow_size, "npc")), 
                          curvature = compute_curvature (x = start_end_df[1, "long"], 
                                                         y = start_end_df[1, "lat"], 
                                                         xend = start_end_df[5, "long"], 
                                                         yend = start_end_df[5, "lat"])
                          ) + 
               geom_curve(aes( x = start_end_df[1, "long"], y = start_end_df[1, "lat"], 
                               xend = start_end_df[6, "long"], yend = start_end_df[6, "lat"]), 
                          colour = "black", data = start_end_df,
                          arrow = arrow(length = unit(arrow_size, "npc")), 
                          curvature = compute_curvature (x = start_end_df[1, "long"], 
                                                         y = start_end_df[1, "lat"], 
                                                         xend = start_end_df[6, "long"], 
                                                         yend = start_end_df[6, "lat"])
                          ) + 
               geom_curve( aes( x = start_end_df[1, "long"], y = start_end_df[1, "lat"], 
                               xend = start_end_df[7, "long"], yend = start_end_df[7, "lat"]), 
                           colour = "black", data = start_end_df,
                           arrow = arrow(length = unit(arrow_size, "npc")), 
                           curvature = compute_curvature (x = start_end_df[1, "long"], 
                                                          y = start_end_df[1, "lat"], 
                                                          xend = start_end_df[7, "long"], 
                                                          yend = start_end_df[7, "lat"])
                          )
  
  return(curr_plot) 
}
##################################################################
plot_the_map_4_web <- function(a_dt, county2, title_p, 
                               target_county_map_info, 
                               most_similar_cnty_map_info, 
                               analog_name){

  curr_plot <- ggplot(a_dt, aes(long, lat, group = group)) + 
               geom_polygon(data = county2, color="black", size=0.05, fill="lightgrey") +
               geom_polygon(aes(fill = analog_freq), colour = rgb(1, 1, .11, .2), size = .01) +
               geom_polygon(data = most_similar_cnty_map_info, color="yellow", 
                            size = .75, fill=NA) +
               geom_polygon(data = target_county_map_info, color="red", 
                            size = .75, fill=NA) +
               borders("state") +
               # ggrepel::geom_text_repel(data = a_dt, aes(x = long, y = lat, label = subregion), 
               #                          fontface = "bold", 
               #                          # nudge_x = c(1, -1.5, 2, 2, -1), 
               #                          # nudge_y = c(0.25, -0.25, 0.5, 0.5, -0.5)
               #                          ) +
               coord_quickmap() +
               guides(fill = guide_colourbar(barwidth = 1, barheight = 20)) + 
               theme(plot.title = element_text(size=18, face="bold"),
                     plot.subtitle = element_text(size=14, face="bold"),
                     plot.margin = unit(c(t=-5, r=5, b=-1, l=5), "pt"),
                     legend.title = element_blank(),
                     legend.position = c(.95, .5), # 
                     legend.background = element_rect(fill = "grey92"),
                     legend.key.size = unit(.7, "line"),
                     legend.text = element_text(size=12, face="bold"),
                     axis.text.x = element_blank(),
                     axis.text.y = element_blank(),
                     axis.ticks.x = element_blank(),
                     axis.ticks.y = element_blank(),
                     axis.title.x = element_blank(),
                     axis.title.y = element_blank()) + 
               ggtitle(title_p, subtitle= paste0("historical analog: ", analog_name))
  return(curr_plot) 
}

plot_fresh_pie <- function(fin_data){
  similarities <- unlist(seperate_1D_similarities(fin_data))
  press_sim <- similarities[1]
  precip_sim <- similarities[2]

  dat = data.frame(count = c(press_sim, (1-press_sim), 
                               precip_sim, (1-precip_sim)),
                   ring = c("pest pressure ring", "pest pressure ring", 
                            "precipitation ring", "precipitation ring"),
                   category = c("variable", "variable complement", 
                                "variable", "variable complement"))
  # compute fractions
  dat %<>% group_by(ring) %>% 
           mutate(fraction = count / sum(count),
                  ymax = cumsum(fraction),
                  ymin = c(0,ymax[1:length(ymax)-1]))
  # Add x limits
  baseNum <- 4
  # numCat <- length(unique(dat$ring))
  dat$xmax <- as.numeric(dat$ring) + baseNum
  dat$xmin = dat$xmax - 1

  p2 = ggplot(dat, aes(fill = category, alpha = ring,
                       ymax = ymax, ymin = ymin, 
                       xmax = xmax, xmin = xmin)) +
       geom_rect(colour = "grey30") +
       coord_polar(theta = "y") +
       xlim(c(0, 6)) +
       theme_bw() +
       theme(panel.grid = element_blank()) +
       theme(axis.text = element_blank()) +
       theme(axis.ticks = element_blank(),
             panel.border = element_blank()) +
       # labs(title = "Customized ring plot") + 
       scale_fill_brewer(palette = "Set1") +
       scale_alpha_discrete(range = c(0.3, 0.8)) + 
       theme(plot.title = element_text(size=18, face="bold"),
             # because its ring, r, l, b are messed up 
             #            (do NOT count on the followings)
             #                   r: is actually left
             #                   l: is actually bottom
             #                   b: is actually right
             # plot.margin = unit(c(t=-10, b=-50, l=-50, r=10), "pt"), # almost good
             # plot.margin = unit(c(t=-10, b=-10, l=-30, r=0), "pt"), # better
             plot.margin = unit(c(t=-10, b=-10, l=-30, r=0), "pt"),
             panel.grid=element_blank(),
             legend.spacing.x = unit(.2, 'pt'),
             legend.title = element_blank(),
             legend.position = "none",
             legend.key.size = unit(1.6, "line"),
             legend.text = element_blank())
  return(p2)
}

plot_the_pie_4_web <- function(DT, titl, subtitle){
  pp <- ggplot(DT, aes(fill=category, ymax=ymax, ymin=ymin, xmax=4, xmin=3)) +
        geom_rect(colour="grey30") +
        coord_polar(theta="y") +
        xlim(c(0, 4)) +
        theme(plot.title = element_text(size=18, face="plain"), 
              plot.subtitle = element_text(size=14, face="plain"),
              plot.margin = unit(c(t=1, b=80, l=10, r=55), "pt"),
              panel.grid=element_blank(),
              legend.spacing.x = unit(.2, 'pt'),
              legend.title = element_blank(),
              legend.position = "none",
              legend.key.size = unit(1.6, "line"),
              legend.text = element_blank()) +
        theme(legend.text = element_blank()) + 
        theme(axis.text = element_blank()) + 
        theme(axis.title=element_blank()) + 
        theme(axis.ticks = element_blank()) # + 
        # labs(title=titl, subtitle= paste0("historical analog: ", subtitle)) +
        # annotate("text", x = 0, y = 0, colour = "red", size = 8,
        #          label = paste0(as.integer(DT[1,2]), "/", as.integer(DT[1,2] + DT[2,2]))) 
  return(pp)
}

plot_the_pie <- function(DT, titl, subtitle){
  pp <- ggplot(DT, aes(fill=category, ymax=ymax, ymin=ymin, xmax=4, xmin=3)) +
        geom_rect(colour="grey30") +
        coord_polar(theta="y") +
        xlim(c(0, 4)) +
        theme(plot.title = element_text(size=20, face="bold"), 
              plot.margin = unit(c(t=0, r=-1, b=1, l=-1), "pt"),
              panel.grid=element_blank(),
              legend.spacing.x = unit(.2, 'pt'),
              legend.title = element_blank(),
              legend.position = "none",
              legend.key.size = unit(1.6, "line"),
              legend.text = element_blank()) +
        theme(legend.text = element_blank()) + 
        theme(axis.text = element_blank()) + 
        theme(axis.title=element_blank()) + 
        theme(axis.ticks = element_blank()) + # + labs(title=titl)
        annotate("text", x = 0, y = 0, colour = "red", size = 8,
                 label = paste0(as.integer(DT[1,2]), "/", as.integer(DT[1,2] + DT[2,2]), 
                                "\n",
                                "most similar to ", "\n", 
                                subtitle)) 
  return(pp)
}
##########################################
##########################################

plot_the_1D_densities <- function(data_dt, dens_T, subT){
  color_ord = c("red", "dodgerblue") #,  "olivedrab4", grey47

  the_theme <- theme(plot.title = element_text(size=20, face="bold"),
                     plot.margin = unit(c(t=.5, r=.5, b=0.5, l=0.5), "pt"),
                     legend.spacing.x = unit(0.4, 'cm'),
                     legend.title = element_blank(),
                     legend.position = "bottom",
                     legend.key.size = unit(1, "line"),
                     legend.text = element_text(size=20, face="plain"),
                     legend.margin = margin(t=.5, r=0, b=.1, l=0, unit = 'cm'),
                     axis.ticks.x = element_blank(),
                     axis.ticks.y = element_blank(),
                     axis.text.x = element_text(size=15, face="bold", color="black"),
                     axis.text.y = element_text(size=15, face="bold", color="black"),
                     axis.title.x = element_text(size=20, face="bold", color="black",
                                                 margin = margin(t=15, r=0, b=0, l=0)),
                     axis.title.y = element_text(size=20, face="bold", color="black", 
                                                 margin = margin(t=0, r=15, b=0, l=0)))

  if ("CumDDinF_Aug23" %in% colnames(data_dt)){
    x_variable_1 <- "CumDDinF_Aug23"
    x_variable_2 <- "yearly_precip"
    
   } else {
    x_variable_1 <- "mean_CumDDinF_Aug23"
    x_variable_2 <- "mean_yearly_precip"
  }
  x_lab_1 <- "Cum. DD (in F) by Aug. 23"
  x_lab_2 <- "annual precip."

  DD_plt <- ggplot(data_dt, aes(x = get(x_variable_1), fill=model, color=model)) +
            geom_density(alpha = 0.1) + 
            scale_color_manual(values=color_ord) + 
            # scale_fill_discrete(guide = guide_legend()) + 
            guides(colour = guide_legend(reverse = TRUE), fill=guide_legend(reverse = TRUE)) + 
            xlab(x_lab_1) +
            ggtitle(label = dens_T, subtitle= paste0( "historical analog: ", subT)) + 
            the_theme 

  preip_plt <- ggplot(data_dt, aes(x = get(x_variable_2), fill=model, color=model)) +
               geom_density(alpha = 0.1) + 
               scale_color_manual(values=color_ord) + 
               # scale_fill_discrete(guide = guide_legend()) + 
               guides(colour = guide_legend(reverse = TRUE), fill=guide_legend(reverse = TRUE)) + 
               xlab(x_lab_1) +
               ggtitle(label = dens_T, subtitle= paste0( "historical analog: ", subT)) + 
               the_theme 

  
  densities <- ggarrange(plotlist = list(DD_plt, preip_plt),
                         ncol = 2, nrow = 1, common.legend=TRUE,
                         legend="bottom")
  return (densities)
}

plot_the_contour <- function(data_dt, con_title, con_subT){ # , v_line_quantiles=c(0.1, 0.9)
  the_theme <- theme(# plot.title = element_text(size=20, face="bold"),
                     plot.margin = unit(c(t=.1, r=.5, b=0.1, l=-3.5), "pt"),
                     legend.spacing.x = unit(0.4, 'cm'),
                     legend.title = element_text(size=20, face="plain"),
                     legend.position = "left",
                     legend.key.size = unit(1, "line"),
                     legend.text = element_text(size=20, face="plain"),
                     legend.margin = margin(t=.5, r=0, b=.1, l=0, unit = 'cm'),
                     axis.ticks.x = element_blank(),
                     axis.ticks.y = element_blank(),
                     axis.text.x = element_text(size=15, face="bold", color="black"),
                     axis.text.y = element_text(size=15, face="bold", color="black"),
                     axis.title.x = element_text(size=20, face="bold", color="black",
                                                 margin = margin(t=15, r=0, b=0, l=0)),
                     axis.title.y = element_text(size=20, face="bold", color="black", 
                                                 margin = margin(t=0, r=15, b=0, l=0)))

  y_lab <- "annual precip. (mm)"
  x_lab <- "pest pressure" # Cum. DD (F) by Aug 23
   
  if ("CumDDinF_Aug23" %in% colnames(data_dt)){
  
    x_variable <- "CumDDinF_Aug23"
    y_variable <- "yearly_precip"
    } else {
    x_variable <- "mean_CumDDinF_Aug23"
    y_variable <- "mean_yearly_precip"
  }
    
  # plot with red fill
  p1 <- ggplot(data = data_dt, aes(x = get(x_variable), y = get(y_variable), color = as.factor(model))) +
        ylab(y_lab) + xlab(x_lab) + 
        stat_density2d(aes(fill = ..level..), alpha = 0.3, geom = "polygon") +
        scale_fill_continuous(low = "grey", high = "red", space = "Lab", name = "modeled") +
        scale_colour_discrete(guide = FALSE) +
        the_theme 

  # plot with blue fill
  p2 <- ggplot(data = data_dt, aes(x = get(x_variable), y = get(y_variable), color = as.factor(model))) +
        stat_density2d(aes(fill = ..level..), alpha = 0.3, geom = "polygon") +
        scale_fill_continuous(low = "grey", high = "blue", space = "Lab", name = "observed") +
        scale_colour_discrete(guide = FALSE) +
        the_theme 

  # grab plot data
  pp1 <- ggplot_build(p1)
  pp2 <- ggplot_build(p2)$data[[1]]

  # replace red fill colours in pp1 with blue colours from pp2 when group is 2
  pp1$data[[1]]$fill[grep(pattern = "^2", pp2$group)] <- pp2$fill[grep(pattern = "^2", pp2$group)]

  # build plot grobs
  grob1 <- ggplot_gtable(pp1)
  grob2 <- ggplotGrob(p2)

  # build legend grobs
  leg1 <- gtable_filter(grob1, "guide-box") 
  leg2 <- gtable_filter(grob2, "guide-box") 
  leg <- gtable:::rbind_gtable(leg1[["grobs"]][[1]],  leg2[["grobs"]][[1]], "first")

  # replace legend in 'red' plot
  grob1$grobs[grob1$layout$name == "guide-box"][[1]] <- leg
  # cowplot::ggdraw(grob1)
  return(grob1)
}

plot_the_contour_stop_working <- function(data_dt, con_title, con_subT, vert_L_type, 
                                          v_line_quantiles=c(0.1, 0.9)){
  the_theme <- theme(# plot.title = element_text(size=20, face="bold"),
                     plot.margin = unit(c(t=0, r=0, b=-2, l=0), "pt"),
                     legend.spacing.x = unit(0.4, 'cm'),
                     legend.title = element_blank(),
                     legend.position = "bottom",
                     legend.key.size = unit(1, "line"),
                     legend.text = element_text(size=20, face="plain"),
                     legend.margin = margin(t=.5, r=0, b=.1, l=0, unit = 'cm'),
                     axis.ticks.x = element_blank(),
                     axis.ticks.y = element_blank(),
                     axis.text.x = element_text(size=15, face="bold", color="black"),
                     axis.text.y = element_text(size=15, face="bold", color="black"),
                     axis.title.x = element_text(size=20, face="bold", color="black",
                                                 margin = margin(t=10, r=0, b=0, l=0)),
                     axis.title.y = element_text(size=20, face="bold", color="black", 
                                                 margin = margin(t=0, r=10, b=0, l=0)))

  y_lab <- "annual precip.(mm)"
  x_lab <- "pest pressure" # Cum. DD (F) by Aug 23

  if (vert_L_type == "historical"){
     line_color <- "springgreen4"
     vertical_dt <- data_dt %>% filter(model == "observed") %>% data.table()
     } else {
     line_color <- "red"
     vertical_dt <- data_dt %>% filter(model != "observed") %>% data.table()
  }
   
  if ("CumDDinF_Aug23" %in% colnames(data_dt)){
     vert <- quantile(vertical_dt$CumDDinF_Aug23, probs=v_line_quantiles)
     horiz <- quantile(vertical_dt$yearly_precip, probs=v_line_quantiles)
     x_variable <- "CumDDinF_Aug23"
     y_variable <- "yearly_precip"
     } else {
     vert <- quantile(vertical_dt$mean_CumDDinF_Aug23, probs=v_line_quantiles)
     horiz <- quantile(vertical_dt$mean_yearly_precip, probs=v_line_quantiles)
     x_variable <- "mean_CumDDinF_Aug23"
     y_variable <- "mean_yearly_precip"
  }
  
  contour_plt <- ggplot(data_dt, aes(x = get(x_variable), y = get(y_variable))) + 
                 # geom_point() + 
                 # geom_density_2d() + 
                 # xlim(800, 4700) + ylim(30, 2700) +
                 ylab(y_lab) + xlab(x_lab) + 
                 stat_density_2d(aes(fill = stat(level), colour = model), 
                                 alpha = .4, contour = TRUE, geom = "polygon") + 
                 scale_fill_viridis_c(guide = FALSE) + 
                 # geom_vline(xintercept = vert[1], color=line_color, size=.5) + 
                 # geom_vline(xintercept = vert[2], color=line_color, size=.5) + 
                 # geom_hline(yintercept = horiz[1], color=line_color, size=.5) + 
                 # geom_hline(yintercept = horiz[2], color=line_color, size=.5) + 
                 the_theme

  return(contour_plt)
}

plot_the_map <- function(a_dt, county2, title_p, 
                         target_county_map_info, 
                         most_similar_cnty_map_info, 
                         analog_name){
    curr_plot <- ggplot(a_dt, aes(long, lat, group = group)) + 
                 geom_polygon(data = county2, fill="lightgrey") +
                 geom_polygon(aes(fill = analog_freq), colour = rgb(1, 1, .11, .2), size = .01) +
                 geom_polygon(data = most_similar_cnty_map_info, color="yellow", size = .75, fill=NA) +
                 geom_polygon(data = target_county_map_info, color="red", size = .75, fill=NA) +
                 borders("state") +
                 coord_quickmap() + 
                 guides(fill = guide_colourbar(barwidth = 1, barheight = 20)) + 
                 theme(plot.title = element_text(size=20, face="bold"),
                       plot.subtitle = element_text(size=15, face="bold"),
                       plot.margin = unit(c(t=1, r=1, b=.5, l=0), "pt"),
                       legend.title = element_blank(),
                       legend.position = c(.95, .3), # 
                       legend.background = element_rect(fill = "grey92"),
                       legend.key.size = unit(1, "line"),
                       legend.text = element_text(size=12, face="bold"),
                       # legend.margin = margin(t=.5, r=0, b=1, l=0, unit = 'cm'),
                       axis.text.x = element_blank(),
                       axis.text.y = element_blank(),
                       axis.ticks.x = element_blank(),
                       axis.ticks.y = element_blank(),
                       axis.title.x = element_blank(),
                       axis.title.y = element_blank()) + 
                 ggtitle(title_p, subtitle= paste0("historical analog: ", analog_name))
    return(curr_plot) 
}

plot_the_pie_all_possible <- function(dat, titl){
  pp <- ggplot(DT, aes(fill=category, ymax=ymax, ymin=ymin, xmax=4, xmin=3)) +
        geom_rect(colour="grey30") +
        coord_polar(theta="y") +
        xlim(c(0, 4)) +
        theme(plot.title = element_text(size=16, face="bold"), 
              panel.grid=element_blank(),
              axis.text = element_blank(),
              axis.ticks = element_blank(),
              legend.spacing.x = unit(.2, 'pt'),
              legend.title = element_blank(),
              legend.position = "bottom",
              legend.key.size = unit(1.6, "line"),
              legend.text = element_text(size=26)) +
        labs(title=titl) + 
        annotate("text", x = 0, y = 0, 
                 label = paste(as.integer(DT[1,2]), as.integer(DT[1,2] + DT[2,2]), sep="/"))
  return(pp)
}

plot_the_pie_Q2 <- function(dat, titl){
  pp <- ggplot(DT, aes(fill=category, ymax=ymax, ymin=ymin, xmax=4, xmin=3)) +
        geom_rect(colour="grey30") +
        coord_polar(theta="y") +
        xlim(c(0, 4)) +
        theme(plot.title = element_text(size=16, face="bold"), 
              panel.grid=element_blank(),
              axis.text = element_blank(),
              axis.ticks = element_blank(),
              legend.spacing.x = unit(.2, 'pt'),
              legend.title = element_blank(),
              legend.position = "bottom",
              legend.key.size = unit(1.6, "line"),
              legend.text = element_text(size=26)) +
        labs(title=titl) + 
        annotate("text", x = 0, y = 0, 
                 label = paste(as.integer(DT[1,2]), as.integer(DT[1,2] + DT[2,2]), sep="/")) 
  return(pp)
}

plot_the_pie_Q3 <- function(dat, titl){
  pp <- ggplot(DT, aes(fill=category, ymax=ymax, ymin=ymin, xmax=4, xmin=3)) +
        geom_rect(colour="grey30") +
        coord_polar(theta="y") +
        xlim(c(0, 4)) +
        theme(plot.title = element_text(size=16, face="bold"), 
              panel.grid=element_blank(),
              axis.text = element_blank(),
              axis.ticks = element_blank(),
              legend.spacing.x = unit(.2, 'pt'),
              legend.title = element_blank(),
              legend.position = "bottom",
              legend.key.size = unit(1.6, "line"),
              legend.text = element_text(size=26)) +
        labs(title=titl) + 
        annotate("text", x = 0, y = 0, 
                 label = paste(as.integer(DT[1,2]), as.integer(DT[1,2] + DT[2,2]), sep="/")) 
  return(pp)
}

##################################################################################
##################################################################################
##################################################################################

plot_100_NN_geo_map <- function(NNs, dists, sigmas, use_sigma=T){
  # For a given location, i.e. a vector,
  # plot the geographical map of 100 NNs
  # based on color
  # input: NNs: data frame of nearest neighbors
  #      dists: distances to the location of interest
  #     sigmas: sigma_dissimilarity between location of interest and other locations
  #  use_sigma: Wheter to use sigma_diss or distances as color codes
  # 
  # output: geographical map of ONE location of interest and its analogs
  #
  year_of_int <- NNs$year
  location_of_int <- NNs$location
  location_of_int <- c(unlist(strsplit(location_of_int, "_"))[1], 
                       unlist(strsplit(location_of_int, "_"))[2])
  location_of_int <- as.numeric(location_of_int)
  
  analogs <- NNs[, seq(2, ncol(NNs_int), 2)]
  
  analogs <- within(analogs, remove(location))
  dists <- within(dists, remove(year, location))
  sigmas <- within(sigmas, remove(year, location))

  x <- sapply(analogs, function(x) strsplit(x, "_")[[1]], USE.NAMES=FALSE)
  lat = x[1, ]
  long = x[2, ]
  
  dt <- setNames(data.table(matrix(nrow = length(sigmas), ncol = 4)), 
                            c("lat", "long", "distances", "sigmas"))
  
  dt$lat = as.numeric(lat)
  dt$long = as.numeric(long)
  dt$distances = as.numeric(dists)
  dt$sigmas = as.numeric(sigmas)
  states <- map_data("state")
}






