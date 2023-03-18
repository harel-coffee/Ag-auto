library(ggplot2)
library(data.table)
library(dplyr)

data_dir <- "/Users/hn/Documents/00_GitHub/Ag/Bloom/Vince_Files/Vince_Bloom_DropBox/"
daymet <- read.csv( paste0(data_dir, "daymet_bloom_wDD.csv"), as.is=TRUE)

gala_dayment <- daymet %>% 
                filter(variety == "Gala") %>%
                data.table()

ggplot(gala_dayment, aes(x=cumdd)) + 
geom_density()


# fitted_normal = pnorm(gala_dayment$cumdd, 
#                       mean = mean(gala_dayment$cumGDD), sd = sd(gala_dayment$cumdd),
#                       lower.tail = TRUE)

gala_normal_fit <- MASS::fitdistr(gala_dayment$cumdd, "normal")

# generate mean and standard dev.
gala_normal_fit_estim <- gala_normal_fit$estimate

fitt_norm_dist <- dnorm(x = gala_dayment$cumdd, 
                        mean = gala_normal_fit_estim[1], 
                        sd   = gala_normal_fit_estim[2])

dist_table <- data.table(cumdd = gala_dayment$cumdd, fitted_dist = fitt_norm_dist)

my_theme <- theme(panel.grid.major = element_line(size=0.2),
                  panel.spacing=unit(.5, "cm"),
                  legend.text=element_text(size=6, face="bold"),
                  legend.title = element_blank(),
                  legend.position = c(0.1, 0.1),
                  strip.text = element_text(face="bold", size=8, color="black"),
                  axis.text = element_text(size=8, color="black"),
                  axis.ticks = element_line(color = "black", size = .2),
                  axis.title.x = element_text(face="bold", size=8, 
                                              margin=margin(t=10, r=0, b=0, l=0)),
                  axis.title.y = element_text(face="bold", size=8, 
                                              margin=margin(t=0, r=10, b=0, l=0)),

                  plot.title = element_text(lineheight=.8, face="bold"))

normal_fit_plot <- ggplot(gala_dayment, aes(x=cumdd)) + # aes(cumdd, fitt_norm_dist)
                   geom_density() + 
                   # geom_point(data=dist_table, aes(cumdd, fitt_norm_dist), shape=1, size=2) + 
                   stat_function(fun = dnorm, n = 101, 
                                 args = list(mean = gala_normal_fit_estim[1], sd = gala_normal_fit_estim[2]),
                                 colour = "red",
                                 size = .5) + 
                   xlab("cumulative DD") + 
                   ylab("probability density") + 
                   scale_y_continuous(breaks = c(0, 0.005 , 0.01),
                                      labels = c(0, 0.005 , 0.01),
                                      limits = c(0, 0.011)) +
                   scale_x_continuous(breaks = c(200, 300 , 400, 500),
                                      labels = c(200, 300 , 400, 500),
                                      limits = c(200, 550)) +
                   my_theme

plot_dir = "/Users/hn/Documents/00_GitHub/Ag_papers/Chill_Paper/MajorRevision/figures_4_revision/"
box_width = 4
box_height = 4
ggsave(filename = "Gala_NormalFitDayMet_Dist.png", 
       plot = normal_fit_plot, 
       path = plot_dir, 
       width = box_width, 
       height = box_height, 
       unit="in", 
       dpi=600)



p <- ggplot(A_Gala , aes(sample = cumGDD)) + 
    stat_qq(distribution = stats::qnorm) + 
    stat_qq_line()

ggplot(A_Gala , aes(sample = cumGDD)) + 
    stat_qq(distribution = ordinal::qgumbel) + 
    stat_qq_line()

EnvStats::qqPlot(A_Gala$cumGDD, 
                 distribution = "norm",
                 param.list = list(mean = 0, sd = 1))

EnvStats::qqPlot(A_Gala$cumGDD, 
                 distribution = "norm",
                 param.list = list(mean = 0403.17, sd = 48.93),
                 add.line = TRUE)



EnvStats::qqPlot(A_Gala$cumGDD, dist = "gamma", 
      estimate.params = TRUE, digits = 2, points.col = "blue", 
      add.line = TRUE)


EnvStats::qqPlot(A_Gala$cumGDD, dist = "weibull", 
      estimate.params = TRUE, digits = 2, points.col = "blue", 
      add.line = TRUE)


fitted_normal = pnorm(A_Gala$cumGDD, 
                      mean = mean(A_Gala$cumGDD), sd = sd(A_Gala$cumGDD),
                      lower.tail = TRUE)


ggplot() + 
geom_point(aes(x=A_Gala$cumGDD, y=fitted_normal)) + 
geom_smooth(aes(x=carat, y=price))


