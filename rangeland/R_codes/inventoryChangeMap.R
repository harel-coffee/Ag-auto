library(usmap)
library(ggplot2)
library(colorspace) # for scale_fill_continuous_diverging()
# library(readspss) # for reading pickles .sav
library(haven) # for reading pickles .sav
library(data.table)

# plot US counties like this
plot_usmap(regions = "counties") + 
labs(title = "US Counties",
     subtitle = "This is a blank map of the counties of the United States.") + 
theme(panel.background = element_rect(color = "black", fill = "lightblue"))



base_dir <- "/Users/hn/Documents/01_research_data/RangeLand/Data/"
diff_dir <- paste0(base_dir, "data_4_plot/")
plot_dir <- paste0(base_dir, "plots/")


# Fucking R cannot read this pickled file.
# Used the Python notebook to create the diffs to plot:
# USDA_data <- haven::read_sav(paste0(data_dir, "USDA_data.sav"))

inventory_AbsChange_2002to2017 <- read.csv(paste0(diff_dir, "inventory_AbsChange_2002to2017.csv"))
inventory_PercChange_2002_2017 <- read.csv(paste0(diff_dir, "inventory_PercChange_2002_2017.csv"))

setnames(inventory_AbsChange_2002to2017, old = c('county_fips'), new = c('fips'))
setnames(inventory_PercChange_2002_2017, old = c('county_fips'), new = c('fips'))



low_lim = min(inventory_AbsChange_2002to2017$cattle_cow_beef_inv_change2002to2017)
up_lim =  max(inventory_AbsChange_2002to2017$cattle_cow_beef_inv_change2002to2017)

SoI_abb <- c('AL', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 
             'FL', 'GA', 'HI', 'ID', 'IL', 'IN', 'IA', 
             'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 
             'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 
             'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 
             'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 
             'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 
             'DC')

states <- plot_usmap("states", color = "red", fill = alpha(0.01)) 

the_theme <-  theme(legend.title = element_text(size=10, face="bold"),
                    legend.text = element_text(size=10, face="plain"),
                    axis.text.x = element_blank(),
                    axis.text.y = element_blank(),
                    axis.ticks.x = element_blank(),
                    axis.ticks.y = element_blank(),
                    axis.title.x = element_blank(),
                    axis.title.y = element_blank(),
                    plot.title = element_text(size=15, lineheight=2, face="bold"))

legend_d <- "Abs. change (2002-2017)"
map <- usmap::plot_usmap(data = inventory_AbsChange_2002to2017, 
                         values = "cattle_cow_beef_inv_change2002to2017",
                         labels = TRUE, label_color = "orange", # color = "orange"
                         # include = SoI_abb
                         ) + 
       # labs(title = "US Counties", subtitle = "abs. change (2002-2017)") +
       geom_polygon(data=states[[1]], aes(x=x, y=y, group=group), 
                    color = "yellow", fill = alpha(0.01), linewidth = 0.5) + 
       scale_fill_gradient2(low = "red", mid = "white", high = "blue", midpoint = 0, 
                            space = "Lab", na.value = "grey50", guide = "colourbar", aesthetics = "fill",
                            name=legend_d) +
       scale_colour_continuous_diverging() + 
       the_theme + 
       ggtitle(paste0(legend_d, ", outlier: Tulare County, CA"))
      
map$layers[[2]]$aes_params$size <- 5
print (map)

ggsave("InventoryChange_2002_2017_absVal.pdf", map, 
       path=plot_dir, device="pdf",
       dpi=300, width=15, height=12, unit="in", limitsize = FALSE)

library(dplyr)
inventory_AbsChange_2002to2017_NoOutlier = inventory_AbsChange_2002to2017 %>%
                                           filter(cattle_cow_beef_inv_change2002to2017<20000)


legend_d <- "Abs. change (2002-2017)"
map <- usmap::plot_usmap(data = inventory_AbsChange_2002to2017_NoOutlier, 
                         values = "cattle_cow_beef_inv_change2002to2017",
                         labels = TRUE, label_color = "orange", # color = "orange"
                         # include = SoI_abb
                         ) + 
       # labs(title = "US Counties", subtitle = "abs. change (2002-2017)") +
       geom_polygon(data=states[[1]], aes(x=x, y=y, group=group), 
                    color = "yellow", fill = alpha(0.01), linewidth = 0.5) + 
       scale_fill_gradient2(low = "red", mid = "white", high = "blue", midpoint = 0, 
                            space = "Lab", na.value = "grey50", guide = "colourbar", aesthetics = "fill",
                            name=legend_d) +
       scale_colour_continuous_diverging() + 
       the_theme + 

       ggtitle(legend_d)
      
map$layers[[2]]$aes_params$size <- 5
print (map)

ggsave("InventoryChange_2002_2017_absVal_noOutlier.pdf", map, 
       path=plot_dir, device="pdf",
       dpi=300, width=15, height=12, unit="in", limitsize = FALSE)


legend_d <- "%-wise change (2002-2017)"
map <- usmap::plot_usmap(data = inventory_PercChange_2002_2017, 
                         values = "change_2002_2017_asPerc",
                         labels = TRUE, label_color = "orange", # color = "orange"
                         # include = SoI_abb
                         ) + 
       # labs(title = "US Counties", subtitle = "abs. change (2002-2017)") +
       geom_polygon(data=states[[1]], aes(x=x, y=y, group=group), 
                    color = "yellow", fill = alpha(0.01), linewidth = 0.5) + 
       scale_fill_gradient2(low = "red", mid = "white", high = "blue", midpoint = 0, 
                            space = "Lab", na.value = "grey50", guide = "colourbar", aesthetics = "fill",
                            name=legend_d) +
       scale_colour_continuous_diverging() + 
       the_theme + 
       ggtitle(paste0(legend_d, ", outlier: Tulare County, CA"))
      
map$layers[[2]]$aes_params$size <- 5
print (map)

ggsave("InventoryChange_2002_2017_percent.pdf", map, 
       path=plot_dir, device="pdf",
       dpi=300, width=15, height=12, unit="in", limitsize = FALSE)


inventory_PercChange_2002_2017_noOutlier <- inventory_PercChange_2002_2017 %>%
                                            filter(change_2002_2017_asPerc < 0.21)




legend_d <- "%-wise change (2002-2017)"
map <- usmap::plot_usmap(data = inventory_PercChange_2002_2017_noOutlier, 
                         values = "change_2002_2017_asPerc",
                         labels = TRUE, label_color = "orange", # color = "orange"
                         # include = SoI_abb
                         ) + 
       # labs(title = "US Counties", subtitle = "abs. change (2002-2017)") +
       geom_polygon(data=states[[1]], aes(x=x, y=y, group=group), 
                    color = "yellow", fill = alpha(0.01), linewidth = 0.5) + 
       scale_fill_gradient2(low = "red", mid = "white", high = "blue", midpoint = 0, 
                            space = "Lab", na.value = "grey50", guide = "colourbar", aesthetics = "fill",
                            name=legend_d) +
       scale_colour_continuous_diverging() + 
       the_theme + 

       ggtitle(legend_d)
      
map$layers[[2]]$aes_params$size <- 5
print (map)

ggsave("InventoryChange_2002_2017_percentNoOutlier.pdf", map, 
       path=plot_dir, device="pdf",
       dpi=300, width=15, height=12, unit="in", limitsize = FALSE)

inventory_PercChange_2002_2017 %>%
filter(change_2002_2017_asPerc>0.2)



