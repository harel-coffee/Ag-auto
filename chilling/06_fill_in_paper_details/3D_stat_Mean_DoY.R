rm(list=ls())
library(data.table)
library(dplyr)
library(ggmap)
library(ggplot2)
library(plotly)
library(plot3D)
options(digit=9)
options(digits=9)

source_path_core = "/Users/hn/Documents/00_GitHub/Ag/chilling/chill_core.R"
source_path_plot = "/Users/hn/Documents/00_GitHub/Ag/chilling/chill_plot_core.R"
source(source_path_plot)
source(source_path_core)

########################################################################################






########################################################################################
data_dir <- "/Users/hn/Documents/01_research_data/bloom/sensitivity_4_chill_paper/"


fullbloom_50percent <- data.table(readRDS(paste0(data_dir, "fullbloom_50percent.rds")))


ict <- c("Omak", "Yakima", "Walla Walla", "Eugene")
TP <- c("2026-2050", "2051-2075", "2076-2099")
ems <- c("RCP 8.5", "RCP 4.5")

fullbloom_50percent$city <- factor(fullbloom_50percent$city, levels = ict, order=TRUE)
fullbloom_50percent$time_period <- factor(fullbloom_50percent$time_period, levels = TP, order=TRUE)
fullbloom_50percent$emission <- factor(fullbloom_50percent$emission, levels = ems, order=TRUE)
df_bloom <- fullbloom_50percent %>%
            filter(time_period == "2076-2099") %>%
            data.table()

plot_DoY_Dist_surface <- function(dt, ct, em){
  dt <- dt[, .(median_over_years = median(dayofyear)), 
             by = c("model", "emission", "fruit_type", 
                    "start_accum_date", "dist_mean",
                    "city", "time_period")]

  dt <- dt[, .(mean_over_models = median(median_over_years)), 
               by = c("emission", "fruit_type", 
                      "start_accum_date", "dist_mean",
                      "city", "time_period")]

  dt <- dt %>% 
        filter(emission == em & city ==ct) %>% 
        data.table()

  #######
  #######   Form the matrix
  #######
  x <- sort(unique(dt$start_accum_date))
  y <- sort(unique(dt$dist_mean))

  z <- matrix(, nrow = length(x), ncol = length(y))

  for (row in 1:length(x)){
     for (col in 1:length(y)){
       z[row, col] <- dt[start_accum_date==x[row] & dist_mean==y[col]]$mean_over_models
     }
  }

  # fig <- plot_ly( type = 'surface',
  #                 contours = list(
  #                 x = list(show = TRUE, start = 1.5, end = 2, size = 0.04, color = 'white'),
  #                 z = list(show = TRUE, start = 0.5, end = 0.8, size = 0.05)),
  #                 x = ~ x,
  #                 y = ~ y,
  #                 z = ~ z)

  # Create lists for axis properties
  f1 <- list(family = "Arial, sans-serif",
  	         size = 18,
  	         color = "c") # lightgrey

  f2 <- list( family = "Arial, sans-serif", #"Old Standard TT, serif",
              size = 14,
              color = "c") # "#ff9999"

  axis <- list( titlefont = f1, tickfont = f2, showgrid = T)
 
  plot_ly(x = y, y = x, z = z, type="surface") %>% 
  layout(# title = "Layout options in a 3d scatter plot",
         scene = list( yaxis = c(list(title = "heat accumulation start DoY"), axis),
                       xaxis = c(list(title = "distribution mean"), axis),
                       zaxis = c(list(title = "50% bloom DoY"), axis),
                       camera = list(eye = list(x = 2, y = 2, z = 2))
                       )
          )
}

plot_DoY_Dist_mesh <- function(dt, ct, em){
  dt <- dt[, .(median_over_years = median(dayofyear)), 
             by = c("model", "emission", "fruit_type", 
                    "start_accum_date", "dist_mean",
                    "city", "time_period")]

  dt <- dt[, .(mean_over_models = median(median_over_years)), 
               by = c("emission", "fruit_type", 
                      "start_accum_date", "dist_mean",
                      "city", "time_period")]

  dt <- dt %>% 
        filter(emission == em & city ==ct) %>% 
        data.table()

  #######
  #######   Form the matrix
  #######
  x <- sort(unique(dt$start_accum_date))
  y <- sort(unique(dt$dist_mean))

  z <- matrix(, nrow = length(x), ncol = length(y))

  for (row in 1:length(x)){
     for (col in 1:length(y)){
       z[row, col] <- dt[start_accum_date==x[row] & dist_mean==y[col]]$mean_over_models
     }
  }

  persp3D_plot <- persp3D(x = x, y = y, z = z, 
                          # theta = 230, phi = 40, 
                          theta = 225, phi = 30, 
                          # theta = 150, phi = 20, 
                          # xlim = c(0, 43),
                          # ylim = c(326, 546),
                          axes=TRUE, scale=3, box=TRUE, 
                          nticks = 8, expand = .4,
                          ticktype = "detailed", 
                          xlab = "heat accumulation start DoY", 
                          ylab = "distribution mean", 
                          zlab = "50% bloom DoY")

}


#############################################
df_bloom <- df_bloom %>% 
            filter(dist_mean >= 435) %>% 
            data.table()

df_bloom <- df_bloom %>% 
            filter(dist_mean <= 543) %>% 
            data.table()

dt = df_bloom 
ct = "Eugene"
em = "RCP 4.5"

plot_DoY_Dist_surface( dt = df_bloom, ct = "Eugene", em="RCP 4.5")
plot_DoY_Dist_mesh(dt = df_bloom, ct = "Eugene", em="RCP 4.5")


#########################
dt = df_bloom 
ct = "Eugene"
em = "RCP 8.5"

plot_DoY_Dist_surface(dt=df_bloom, ct = "Eugene", em="RCP 8.5")
plot_DoY_Dist_mesh(dt=df_bloom, ct = "Eugene", em="RCP 8.5")

#############################################



#############################################

# legendtitle <- list(yref='paper',xref="paper",y=1.05,x=1.1, text="DoY",showarrow=F)

# plot_ly(x = ~dt$start_accum_date, 
#         y = ~dt$dist_mean, 
#         z = ~dt$mean_over_models, 
#         type = 'mesh3d',
#         intensity = seq(min(dt$mean_over_models), max(dt$mean_over_models), length = length(dt$mean_over_models)),
#         # color = seq(0, 1, length = length(dt$start_accum_date)),
#         colors = colorRamp(rainbow((dt$mean_over_models)))
#         ) %>%
# layout(# title = "Layout options in a 3d scatter plot",
#         annotations=legendtitle ,
#         scene = list( yaxis = c(list(title = "heat accumulation start DoY"), axis),
#                       xaxis = c(list(title = "distribution mean"), axis),
#                       zaxis = c(list(title = "50% bloom DoY"), axis),
#                       camera = list(eye = list(x = 2, y = 2, z = 2))
#                       )
#         )


################################################
####
####
####
################################################

# heat_Feb15 <- data.table(readRDS(paste0(data_dir, "heat_Feb15_unmelt.rds")))
# #
# # dist_mean does not matter
# #
# heat_Feb15 <- within(heat_Feb15, remove(dist_mean))
# heat_Feb15 <- unique(heat_Feb15)

# heat_Feb15$city <- factor(heat_Feb15$city, levels = ict, order=TRUE)
# heat_Feb15$time_period <- factor(heat_Feb15$time_period, levels = TP, order=TRUE)
# heat_Feb15$emission <- factor(heat_Feb15$emission, levels = ems, order=TRUE)

# heat_Feb15 <- heat_Feb15 %>%
#               filter(year >= 2076)%>%
#               data.table()


# df_heat <- heat_Feb15 %>%
#            filter(time_period == "2076-2099") %>%
#            data.table()

# plot_heat_acum_by_Feb15_not_needed_really_distrubution_does_not_matter <- function(dt){
#   dt <- dt[, .(median_over_years = median(vert_Cum_dd)), 
#              by = c("model", "emission", "col_type", 
#                     "start_accum_date", "dist_mean",
#                     "city", "time_period")]

#   dt <- dt[, .(mean_over_models = median(median_over_years)), 
#                by = c("emission", "col_type", 
#                       "start_accum_date", "dist_mean",
#                       "city", "time_period")]

#   dt <- dt %>% 
#         filter(emission == "RCP 8.5" & city == "Eugene") %>% 
#         data.table()

#   #######
#   #######   Form the matrix
#   #######
#   x <- sort(unique(dt$start_accum_date))
#   y <- sort(unique(dt$dist_mean))

#   z <- matrix(, nrow = length(x), ncol = length(y))

#   for (row in 1:length(x)){
#     for (col in 1:length(y)){
#       z[row, col] <- dt[start_accum_date==x[row] & dist_mean==y[col]]$mean_over_models
#     }
#   }

#   # fig <- plot_ly( type = 'surface',
#   #                 contours = list(
#   #                 x = list(show = TRUE, start = 1.5, end = 2, size = 0.04, color = 'white'),
#   #                 z = list(show = TRUE, start = 0.5, end = 0.8, size = 0.05)),
#   #                 x = ~ x,
#   #                 y = ~ y,
#   #                 z = ~ z)

#   # Create lists for axis properties
#   f1 <- list(
#     family = "Arial, sans-serif",
#     size = 18,
#     color = "c") # lightgrey

#   f2 <- list(
#     family = "Arial, sans-serif", #"Old Standard TT, serif",
#     size = 14,
#     color = "c") # "#ff9999"

#   axis <- list(
#     titlefont = f1,
#     tickfont = f2,
#     showgrid = T
#   )

#   fig <- plot_ly(x = x, y = y, z = z) %>% 
#          add_surface() %>% 
#          layout(# title = "Layout options in a 3d scatter plot",
#                 scene = list( xaxis = c(list(title = "heat accumulation start date"), axis),
#                               yaxis = c(list(title = "distribution mean"), axis),
#                               zaxis = c(list(title = "VGDD by Feb. 15"), axis),
#                               camera = list(eye = list(x = 2, y = 2, z = 2))
#                            )
#                )
# }


# cloudy_VGDD <- function(d1, colname="median_over_years", fil="GDD"){
#   cls = "darkorchid"
  
#   ggplot(d1, aes(x=start_accum_date, y=get(colname), fill=fil)) +
#   labs(x = "accumulation start date", y = "accumulated GDD") + #, fill = "Climate Group"
#   facet_grid(. ~ emission ~ city) + # scales = "free"
#   stat_summary(geom="ribbon", fun.y=function(z) { quantile(z,0.5) }, 
#                               fun.ymin=function(z) { quantile(z,0) }, 
#                               fun.ymax=function(z) { quantile(z,1) }, 
#                alpha=0.2) +

#   stat_summary(geom="ribbon", fun.y=function(z) {quantile(z,0.5) }, 
#                fun.ymin=function(z) { quantile(z,0.1) }, 
#                fun.ymax=function(z) { quantile(z,0.9) }, alpha=0.4) +

#   stat_summary(geom="ribbon", fun.y=function(z) { quantile(z,0.5) }, 
#                               fun.ymin=function(z) { quantile(z,0.25) }, 
#                               fun.ymax=function(z) { quantile(z,0.75) }, 
#                alpha=0.8) +

#   stat_summary(geom="line", fun.y=function(z) {quantile(z,0.5) }, 
#                size = 1) + 
#   scale_color_manual(values = cls) +
#   scale_fill_manual(values = cls) +
#   scale_x_continuous(breaks = sort(unique(d1$start_accum_date))) +
#   theme(panel.grid.major = element_line(size=0.2),
#         panel.spacing=unit(.5, "cm"),
#         legend.text=element_text(size=18, face="bold"),
#         legend.title = element_blank(),
#         legend.position = "bottom",
#         strip.text = element_text(face="bold", size=16, color="black"),
#         axis.text = element_text(size=16, color="black"), # face="bold",
#         axis.text.x = element_text(hjust = 1),
#         axis.ticks = element_line(color = "black", size = .2),
#         axis.title.x = element_text(size=18,  face="bold", 
#                                     margin=margin(t=10, r=0, b=0, l=0)),
#         axis.title.y = element_text(size=18, face="bold",
#                                     margin=margin(t=0, r=10, b=0, l=0)),
#         plot.title = element_text(lineheight=.8, face="bold", size=20)
#         )
# }

# heat_Feb15 <- heat_Feb15[, .(median_over_years = median(vert_Cum_dd)), 
#                              by = c("model", "emission", 
#                              "start_accum_date",
#                              "city", "time_period")]

# GDD_plt <- cloudy_VGDD(d1=heat_Feb15)

# plot_dir <- "/Users/hn/Documents/00_GitHub/Ag_papers/Chill_Paper/figures/GDD_sensitivity/"
# if (dir.exists(plot_dir) == F) {dir.create(path = plot_dir, recursive = T)}

# if (length(unique(heat_Feb15$city)) == 2){
#   W = 10
#   } else if (length(unique(heat_Feb15$city)) == 1){
#   W = 5
# }

# W = 12

# ggsave(plot=GDD_plt,
#        filename = paste0("GDD_plt.png"), 
#        width=W, height=5, units = "in", 
#        dpi=600, device = "png",
#        path=plot_dir)

