library(data.table)
library(dplyr)
library(ggplot2)



data_dir = "/Users/hn/Documents/00_GitHub/Ag/Bloom/Vince_Files/Vince_Bloom_DropBox/"
daymet_bloom = read.csv(paste0(data_dir, "daymet_bloom_wDD.csv"), as.is=TRUE)

daymet_bloom_gala <- daymet_bloom %>% 
                     filter(variety == "Gala") %>%
                     data.table()


###########################
d <- density(daymet_bloom_gala$cumdd) # returns the density data
plot(d) # plots the results

d <- density(daymet_bloom_gala$cumGDD) # returns the density data
plot(d) # plots the results


daymet_bloom_gala_mean <- daymet_bloom_gala %>%
                          group_by(station, year, date) %>%
                          summarise(meancumGDD= mean(cumGDD))%>%
                          data.table()

d <- density(daymet_bloom_gala_mean$meancumGDD) # returns the density data
plot(d) # plots the results


ggplot(daymet_bloom_gala_mean, aes(x=meancumGDD)) + 
geom_density() + 
facet_wrap(~ station)


ggplot(daymet_bloom_gala, aes(x=cumGDD)) + 
geom_density() + 
facet_wrap(~ station)

###########################
fitted_normal = pnorm(daymet_bloom_gala$cumdd, 
                             mean = mean(daymet_bloom_gala$cumGDD), sd = sd(daymet_bloom_gala$cumdd),
                             lower.tail = TRUE)


ggplot() +
geom_qq(aes(sample = daymet_bloom_gala$cumdd))


ggplot() + 
geom_point(aes(x=daymet_bloom_gala$cumdd, y=fitted_normal)) + 
geom_smooth(aes(x=daymet_bloom_gala$cumdd, y=fitted_normal))


ggplot(daymet_bloom_gala , aes(sample = cumdd)) + 
     stat_qq(distribution = stats::qnorm) + 
     stat_qq_line()

# plot(density(daymet_bloom_gala$cumdd))


##########################################################################################
gala_normal_fit <- MASS::fitdistr(daymet_bloom_gala$cumdd, "normal")

# generate mean and standard dev.
gala_normal_fit_estim <- gala_normal_fit$estimate

fitt_norm_dist <- dnorm(x = daymet_bloom_gala$cumdd, 
                        mean = gala_normal_fit_estim[1], 
                        sd   = gala_normal_fit_estim[2])


# plot(x = daymet_bloom_gala$cumdd, y = fitt_norm_dist)

dist_table <- data.table(cumdd = daymet_bloom_gala$cumdd, fitted_dist = fitt_norm_dist)

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

normal_fit_plot <- ggplot() + # aes(cumdd, fitt_norm_dist)
                   geom_point(data=dist_table, aes(cumdd, fitt_norm_dist), shape=1, size=2) + 
                   stat_function(fun = dnorm, n = 101, 
                                 args = list(mean = gala_normal_fit_estim[1], sd = gala_normal_fit_estim[2]),
                                 colour = "red",
                                 size = .5) + 
                   xlab("cumulative DD") + 
                   ylab("probability density") + 
                   scale_y_continuous(breaks = c(0, 0.005 , 0.01),
                                      labels = c(0, 0.005 , 0.01),
                                      limits = c(0, 0.01)) +
                   scale_x_continuous(breaks = c(200, 300 , 400, 500),
                                      labels = c(200, 300 , 400, 500),
                                      limits = c(200, 550)) +
                   my_theme


plot_dir = "/Users/hn/Documents/00_GitHub/Ag_papers/Chill_Paper/MajorRevision/figures_4_revision/"
box_width = 4
box_height = 4
ggsave(filename = "Gala_NormalFitDayMet.png", 
       plot = normal_fit_plot, 
       path = plot_dir, 
       width = box_width, 
       height = box_height, 
       unit="in", 
       dpi=600)

##########################################################################################

#
# compute frequencies of actual cumdd
#
freq_table <- data.table(table(daymet_bloom_gala$cumdd))
setnames(freq_table, old=c("V1", "N"), new=c("cumdd", "count"))

freq_table$prob <- freq_table$count / length(freq_table$count)


ggplot() + # aes(cumdd, fitt_norm_dist)
geom_point(data=dist_table, aes(cumdd, fitt_norm_dist), shape=1, size=5) + 
# geom_point(data=freq_table, aes(cumdd, prob), shape=1, size=5) + 
stat_function(fun = dnorm, n = 101, 
                  args = list(mean = gala_normal_fit_estim[1], sd = gala_normal_fit_estim[2]),
                  colour = "red",
                  size = 2) + 
xlab("cumulative DD") + 
ylab("probability density") + 
scale_y_continuous(breaks = c(0, 0.005 , 0.01),
                         labels = c(0, 0.005 , 0.01),
                         limits = c(0, 0.01)) +
scale_x_continuous(breaks = c(200, 300 , 400, 500),
                         labels = c(200, 300 , 400, 500),
                         limits = c(200, 550)) +
my_theme





