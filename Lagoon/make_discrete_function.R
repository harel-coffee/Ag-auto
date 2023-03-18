
require(ggplot2)
require(cowplot)

values <- c(0, 1, 2, 5, 10) # this vector is needed not only for the data frame cbar, but also for plotting
group <- letters[1:5]
diff_values <- c(0, diff(values))
cbar_df <- data.frame(x = 1, y = values, diff_values, group,  stringsAsFactors = FALSE)
# that's for the fake legend

iris2 <- iris #don't want to mess with your iris data set
              #I used iris because you hadn't provided data
iris2$cuts <- cut(iris2$Petal.Length, values) #the already offered 'cut-approach' 

p1  <- ggplot(iris2, aes(Sepal.Length, y = Sepal.Width, color = cuts))+ 
          geom_point() +
          scale_color_brewer("", palette = "Reds")

cbar_plot <- ggplot(cbar_df, aes(x, y = diff_values, fill = c(NA, rev(group[2:5])))) + 
  # I had to do implement this 'fill=' workaround 
  # in creating a new vector introducing an NA, 
  # and I had to flip the fills in order to fit to the scale... 
    geom_col(width = 0.1, show.legend = FALSE)  +
    geom_segment(y = values, yend = values, x = 0.9, xend = 1.05) +
    annotate(geom = 'text', x = 0.85, y = values, label = values) +
  # the numbers are quasi-randomly chosen but define the length of your ticks, obviously
    scale_x_continuous(expand = c(1,1)) + 
  # you might need to play around with the expand argument for the width of your legend
    scale_fill_brewer("", palette = "Reds", direction = -1) +  
  # don't know why you have to flip this again... 
    coord_flip() +
    theme_void()

plot_grid(p1, cbar_plot, nrow = 2)


geo_map_of_diffs <- function(dt, col_col, minn, maxx, ttl, subttl){
  ##
  ## This also exist in the core_plot
  ##
  color_limit <- max(abs(minn), abs(maxx))
  x <- sapply(dt$location, 
              function(x) strsplit(x, "_")[[1]], 
              USE.NAMES=FALSE)
  lat <- as.numeric(x[1, ]); long <- as.numeric(x[2, ])
  dt$lat <- lat; dt$long <- long;
  
  states <- map_data("state")
  states_cluster <- subset(states, 
                           region %in% c("washington"))
  WA_counties <- map_data("county", "washington")
  WA_counties <- WA_counties %>% 
                 filter(subregion %in% c("whatcom", "skagit", "snohomish", "island"
                                         #, "okanogan" "chelan"
                                         ))%>% 
                 data.table()
  
  dt %>%
  ggplot() +
  geom_polygon(data = WA_counties, 
               aes(x = long, y = lat, group = group),
               fill = "grey", color = "black") +
  geom_polygon(data=WA_counties, 
               aes(x=long, y=lat, group = group), 
               fill = NA, colour = "grey60", size=.3) + 
  geom_point(aes_string(x = "long", y = "lat", color = col_col), 
             alpha = 1, size=.3) +
  guides(fill = guide_colourbar(barwidth = .1, barheight = 20))+
  scale_color_gradient2(midpoint = 0, mid = "white", 
                        high = muted("blue"), low = muted("red"), 
                        guide = "colourbar", space = "Lab",
                        limit = c(-color_limit, color_limit)) + 
  theme(axis.title.y = element_blank(),
        axis.title.x = element_blank(),
        axis.ticks.y = element_blank(), 
        axis.ticks.x = element_blank(),
        axis.text = element_blank(),
        plot.title = element_text(size = 14, face = "bold"),
        legend.text = element_text(size = 8, face="plain"),
        legend.title = element_blank(),
        legend.position = "top",
        strip.text = element_text(size=14, face="bold"))
  +
  ggtitle(ttl, subtitle=subttl)
  
}

