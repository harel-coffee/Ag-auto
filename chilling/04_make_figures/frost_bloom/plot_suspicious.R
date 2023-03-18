library(data.table)
library(dplyr)
library(ggpubr)

options(digits=9)
options(digit=9)

data_dir <- "/Users/hn/Desktop/Desktop/Ag/check_point/chilling/"
data_dir <- paste0(data_dir, "suspicious_points_not_reaching_threshods/")


"chill_output_data_46.03125_-118.34375.txt"
walla_walla_HadGEM2ES365 <- read.table(file = paste0(data_dir, 
                                                    "walla_walla_HadGEM2ES365.txt"),
                         header = T,
                         colClasses = c("factor", "numeric", "numeric", "numeric",
                                        "numeric", "numeric"))

walla_walla <- walla_walla_HadGEM2ES365 %>% 
               filter(chill_season %in% c("chill_2083-2084", 
                                          "chill_2084-2085",
                                          "chill_2085-2086",
                                          "chill_2086-2087",
                                          "chill_2087-2088")) %>%
               data.table()

walla_walla$dayofyear <- 1
walla_walla <- data.table(walla_walla, key = "chill_season")
walla_walla[, dayofyear := cumsum(dayofyear), by = key(walla_walla)]

five_years <- ggplot(data=walla_walla) + 
              geom_point(aes(x = dayofyear, y = cume_portions), alpha = 0.25, shape = 21, size = .5) + 
              geom_line(aes(x = dayofyear, y = cume_portions, group=chill_season, color=chill_season)) + 
              theme(plot.title = element_text(size=17, face="bold"),
                    plot.margin = margin(t=.2, r=.5, b=.2, l=.2, "cm"),
                    panel.grid.minor = element_blank(),
                    panel.spacing = unit(.5, "cm"),
                    panel.grid.major = element_line(size = 0.1),
                    axis.ticks = element_line(color = "black", size = .2),
                    strip.text.x = element_text(size = 20, face = "bold"),
                    strip.text.y = element_text(size = 20, face = "bold"),
                    axis.text.x = element_text(size = 16, face = "plain", color="black"),
                    axis.text.y = element_text(size = 16, color="black"),
                    axis.title.x = element_blank(),
                    axis.title.y = element_blank()
                    )


plot_dir <- data_dir
ggsave(plot = five_years,
       filename = "walla_walla_75_not_hit.png", 
       width = 8, height=4, units = "in",
       dpi=400, device = "png", path=plot_dir)

