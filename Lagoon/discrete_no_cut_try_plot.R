geo_map_of_diffs <- function(dt, col_col, minn, maxx, ttl, subttl){

  x <- sapply(dt$location,
              function(x) strsplit(x, "_")[[1]], 
              USE.NAMES=FALSE)
  lat <- as.numeric(x[1, ]); long <- as.numeric(x[2, ])
  dt$lat <- lat; dt$long <- long;
  
  states <- map_data("state")
  states_cluster <- subset(states, 
                           region %in% c("washington"))
  WA_counties <- map_data("county", "washington")
  
  dt %>%
  ggplot() +
  geom_polygon(data = states_cluster, 
               aes(x = long, y = lat, group = group),
               fill = "grey", color = "black") +
  geom_polygon(data=WA_counties, 
               aes(x=long, y=lat, group = group), 
               fill = NA, colour = "grey60", size=.3) + 
  geom_point(aes_string(x = "long", y = "lat", color = "tagt_perc"), 
             alpha = 1, size=.3) +
  scale_color_manual(name = "qsec",
                     values = c("(-Inf,-0.5]"= "red4",
                                "(-0.5,-0.2]" = "red",
                                "(-0.2,-0.15]" = "brown1",
                                "(-0.15,-0.1]" = "tomato1",
                                "(-0.1,-0.05]" = "hotpink1",
                                "(-0.05,0]" = "white",
                                "(0,0.05]"  = "white",
                                "(0.05,0.1]" = "lightskyblue",
                                "(0.1,0.15]" = "deepskyblue",
                                "(0.15,0.2]" = "dodgerblue3",
                                "(0.2,0.5]"  = "dodgerblue4",
                                "(0.5,Inf]"  = "blue4"
                                ),
                     # labels = c("<= 17", "17 < qsec <= 19", "> 19")
                     )  + 
  theme(axis.title.y = element_blank(),
        axis.title.x = element_blank(),
        axis.ticks.y = element_blank(), 
        axis.ticks.x = element_blank(),
        axis.text = element_blank(),
        plot.title = element_text(size = 14, face = "bold"),
        legend.text = element_text(size = 8, face="plain"),
        legend.title = element_blank(),
        # legend.justification = c(.93, .9),
        # legend.position = c(.93, .9),
        legend.position = "top",
        strip.text = element_text(size=14, face="bold"))+
  ggtitle(ttl, subtitle=subttl)
  
}