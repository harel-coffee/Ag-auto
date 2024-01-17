rm(list=ls())
library(usmap)
library(ggplot2)
library(colorspace) # for scale_fill_continuous_diverging()
# library(readspss) # for reading pickles .sav
library(haven) # for reading pickles .sav
library(data.table)
library(dplyr)
# plot US counties like this
plot_usmap(regions = "counties") + 
labs(title = "US Counties",
     subtitle = "This is a blank map of the counties of the United States.") + 
theme(panel.background = element_rect(color = "black", fill = "lightblue"))



base_dir <- "/Users/hn/Documents/01_research_data/RangeLand/Data/"
diff_dir <- paste0(base_dir, "data_4_plot/")
plot_dir <- paste0(base_dir, "plots/")


# Fucking R cannot read this pickled file.
# Used the Python notebook (inventory_diff_4_MapinR.ipynb) to create the diffs to plot:
# USDA_data <- haven::read_sav(paste0(data_dir, "USDA_data.sav"))

inventory_AbsChange_2002to2017 <- read.csv(paste0(diff_dir, "inventory_AbsChange_2002to2017.csv"))
inv_PercChangeShare_2002_2017 <- read.csv(paste0(diff_dir, "inv_PercChangeShare_2002_2017.csv"))

setnames(inventory_AbsChange_2002to2017, old = c('county_fips'), new = c('fips'))
setnames(inv_PercChangeShare_2002_2017, old = c('county_fips'), new = c('fips'))


# low_lim = min(inventory_AbsChange_2002to2017$inv_change2002to2017)
# up_lim =  max(inventory_AbsChange_2002to2017$inv_change2002to2017)

SoI_abb <- c('AL', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 
             'FL', 'GA', 'HI', 'ID', 'IL', 'IN', 'IA', 
             'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 
             'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 
             'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 
             'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 
             'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 
             'DC')

states <- plot_usmap("states", color = "red", fill = alpha(0.01))

the_theme <-  theme(legend.title = element_text(size=25, face="bold"),
                    legend.text = element_text(size=25, face="plain"),
                    legend.key.size = unit(1, 'cm'), #change legend key size
                    legend.key.height = unit(1, 'cm'), #change legend key height
                    legend.key.width = unit(1, 'cm'), #change legend key width
                    axis.text.x = element_blank(),
                    axis.text.y = element_blank(),
                    axis.ticks.x = element_blank(),
                    axis.ticks.y = element_blank(),
                    axis.title.x = element_blank(),
                    axis.title.y = element_blank(),
                    plot.title = element_text(size=25, lineheight=2, face="bold"))

legend_d <- "Abs. change (2002-2017)"
map <- usmap::plot_usmap(data = inventory_AbsChange_2002to2017, 
                         values = "inv_change2002to2017",
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
      
map$layers[[2]]$aes_params$size <- 8
print (map)

ggsave("InventoryChange_2002_2017_absVal.pdf", map, 
       path=plot_dir, device="pdf",
       dpi=300, width=15, height=12, unit="in", limitsize = FALSE)


inventory_AbsChange_2002to2017_NoOutlier = inventory_AbsChange_2002to2017 %>%
                                           filter(inv_change2002to2017<20000)

legend_d <- "Abs. change (2002-2017)"
map <- usmap::plot_usmap(data = inventory_AbsChange_2002to2017_NoOutlier, 
                         values = "inv_change2002to2017",
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
      
map$layers[[2]]$aes_params$size <- 8
print (map)

ggsave("InventoryChange_2002_2017_absVal_noOutlier.pdf", map, 
       path=plot_dir, device="pdf",
       dpi=300, width=15, height=12, unit="in", limitsize = FALSE)



legend_d <- "Percentage change (2002-2017)"
map <- usmap::plot_usmap(data = inventory_AbsChange_2002to2017, 
                         values = "inv_change2002to2017_asPerc",
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
      
map$layers[[2]]$aes_params$size <- 8
print (map)

ggsave("percChange_2002_2017.pdf", map, 
       path=plot_dir, device="pdf",
       dpi=300, width=15, height=12, unit="in", limitsize = FALSE)


inventory_PercChange_2002to2017_NoOutlier = inventory_AbsChange_2002to2017 %>%
                                           filter(inv_change2002to2017_asPerc<200)

legend_d <- "Percentage change (2002-2017)"
map <- usmap::plot_usmap(data = inventory_PercChange_2002to2017_NoOutlier, 
                         values = "inv_change2002to2017_asPerc",
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
      
map$layers[[2]]$aes_params$size <- 8
print (map)

ggsave("percChange_2002_2017_noOutlier.pdf", map, 
       path=plot_dir, device="pdf",
       dpi=300, width=15, height=12, unit="in", limitsize = FALSE)

########################################
########################################
########################################
#
# Actual Inventory
#
legend_d <- "Inventory: 2002"
map <- usmap::plot_usmap(data = inventory_AbsChange_2002to2017, 
                         values = "cattle_cow_beef_inven_2002",
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
      
map$layers[[2]]$aes_params$size <- 8
print (map)

ggsave("inven_2002.pdf", map, 
       path=plot_dir, device="pdf",
       dpi=300, width=15, height=12, unit="in", limitsize = FALSE)


legend_d <- "Inventory: 2017"
map <- usmap::plot_usmap(data = inventory_AbsChange_2002to2017, 
                         values = "cattle_cow_beef_inven_2017",
                         labels = TRUE, label_color = "orange", # color = "orange"
                         # include = SoI_abb
                         ) + 
       geom_polygon(data=states[[1]], aes(x=x, y=y, group=group), 
                    color = "yellow", fill = alpha(0.01), linewidth = 0.5) + 
       scale_fill_gradient2(low = "red", mid = "white", high = "blue", midpoint = 0, 
                            space = "Lab", na.value = "grey50", guide = "colourbar", aesthetics = "fill",
                            name=legend_d) +
       scale_colour_continuous_diverging() + 
       the_theme + 

       ggtitle(legend_d)
      
map$layers[[2]]$aes_params$size <- 8
print (map)

ggsave("inven_2017.pdf", map, 
       path=plot_dir, device="pdf",
       dpi=300, width=15, height=12, unit="in", limitsize = FALSE)
########################################
########################################
########################################
###
### Actual shares
###
legend_d <- "%-of national share (2002)"
map <- usmap::plot_usmap(data = inv_PercChangeShare_2002_2017, 
                         values = "inv_2002_asPercShare",
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
       ggtitle(paste0(legend_d))
      
map$layers[[2]]$aes_params$size <- 5
print (map)

ggsave("nationalSharePercent_2002.pdf", map, 
       path=plot_dir, device="pdf",
       dpi=300, width=15, height=12, unit="in", limitsize = FALSE)


legend_d <- "%-of national share (2017)"
map <- usmap::plot_usmap(data = inv_PercChangeShare_2002_2017, 
                         values = "inv_2017_asPercShare",
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
       ggtitle(paste0(legend_d))
      
map$layers[[2]]$aes_params$size <- 5
print (map)

ggsave("nationalSharePercent_2017.pdf", map, 
       path=plot_dir, device="pdf",
       dpi=300, width=15, height=12, unit="in", limitsize = FALSE)

########################################
########################################
########################################
###
###  diff in shares
###
legend_d <- "%-wise change as national share (2002-2017)"
map <- usmap::plot_usmap(data = inv_PercChangeShare_2002_2017, 
                         values = "change_2002_2017_asPercShare",
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

ggsave("NationalShareChange_2002_2017_percent.pdf", map, 
       path=plot_dir, device="pdf",
       dpi=300, width=15, height=12, unit="in", limitsize = FALSE)


inv_PercChangeShare_2002_2017_noOutlier <- inv_PercChangeShare_2002_2017 %>%
                                           filter(change_2002_2017_asPercShare > -0.20)


legend_d <- "%-wise change as national share (2002-2017)"
map <- usmap::plot_usmap(data = inv_PercChangeShare_2002_2017_noOutlier, 
                         values = "change_2002_2017_asPercShare",
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

ggsave("NationalShareChange_2002_2017_percent_NoOutlier.pdf", map, 
       path=plot_dir, device="pdf",
       dpi=300, width=15, height=12, unit="in", limitsize = FALSE)

inv_PercChangeShare_2002_2017 %>%
filter(change_2002_2017_asPercShare>0.2)

##################### All years. 
##################### Panel is not gonna look good! 
##################### So, one at a time


inventory_AbsChange <- read.csv(paste0(diff_dir, "inventory_AbsChange_4panel.csv"))
inv_PercChangeShare <- read.csv(paste0(diff_dir, "inventory_ShareChange_4panel.csv"))

setnames(inventory_AbsChange, old = c('county_fips'), new = c('fips'))
setnames(inv_PercChangeShare, old = c('county_fips'), new = c('fips'))


Abs_plot_dir <- paste0(plot_dir, "abs_change_map/")
if (dir.exists(Abs_plot_dir) == F) {dir.create(path = Abs_plot_dir, recursive = T)}
share_plot_dir <- paste0(plot_dir, "share_change_map/")
if (dir.exists(share_plot_dir) == F) {dir.create(path = share_plot_dir, recursive = T)}

for (a_col in colnames(inventory_AbsChange)[4:length(colnames(inventory_AbsChange))]){
  if (str_detect(a_col, pat="change")){
    s_year = substr(a_col, start = 11, stop = 14)
    e_year = substr(a_col, start = 17, stop = 20)
    if (str_detect(a_col, pat="asPerc")){
       legend_d <- paste0("Percentage change ", s_year, "_", e_year)
      } else {
       legend_d <- paste0("Absolute change ", s_year, "_", e_year)
      }
   } else {
       yr = tail(strsplit(a_col, split = "_")[[1]], 1)
       legend_d <- paste0("Invenory ", yr)

   }

  inventory_AbsChange_copy = copy(inventory_AbsChange)
  inventory_AbsChange_copy <- subset(inventory_AbsChange_copy, select = c("fips", a_col))
  inventory_AbsChange_copy <- inventory_AbsChange_copy %>% drop_na()
  


  
  map <- usmap::plot_usmap(data = inventory_AbsChange_copy, 
                           values = a_col,
                           labels = TRUE, label_color = "orange",
                           ) + 
         geom_polygon(data=states[[1]], aes(x=x, y=y, group=group), 
                      color = "yellow", fill = alpha(0.01), linewidth = 0.5) + 
         scale_fill_gradient2(low = "red", mid = "white", high = "blue", midpoint = 0, 
                              space = "Lab", na.value = "grey50", guide = "colourbar", aesthetics = "fill",
                              name=legend_d) +
         scale_colour_continuous_diverging() + 
         the_theme + 
         ggtitle(legend_d)
      
  map$layers[[2]]$aes_params$size <- 8

  ggsave(paste0(gsub("\ ", "_", legend_d), ".pdf"), map, 
         path=Abs_plot_dir, device="pdf",
         dpi=300, width=15, height=12, unit="in", limitsize = FALSE)
}



##### Changes as in National Share

for (a_col in colnames(inv_PercChangeShare)[4:length(colnames(inv_PercChangeShare))]){
  if (str_detect(a_col, pat="change")){
    s_year = unlist(strsplit(a_col, split = "_"))[2]
    e_year = unlist(strsplit(a_col, split = "_"))[3]
    legend_d <- paste0("Change of share ", s_year, "_", e_year)
   } else {
       yr = unlist(strsplit(a_col, split = "_"))[2]
       legend_d <- paste0("Invenory ", yr, " as percentage share")
   }

  inv_PercChangeShare_copy = copy(inv_PercChangeShare)
  inv_PercChangeShare_copy <- subset(inv_PercChangeShare_copy, select = c("fips", a_col))
  inv_PercChangeShare_copy <- inv_PercChangeShare_copy %>% drop_na()
  
  map <- usmap::plot_usmap(data = inv_PercChangeShare_copy, 
                           values = a_col,
                           labels = TRUE, label_color = "orange",
                           ) + 
         geom_polygon(data=states[[1]], aes(x=x, y=y, group=group), 
                      color = "yellow", fill = alpha(0.01), linewidth = 0.5) + 
         scale_fill_gradient2(low = "red", mid = "white", high = "blue", midpoint = 0, 
                              space = "Lab", na.value = "grey50", guide = "colourbar", aesthetics = "fill",
                              name=legend_d) +
         scale_colour_continuous_diverging() + 
         the_theme + 
         ggtitle(legend_d)
      
  map$layers[[2]]$aes_params$size <- 8

  ggsave(paste0(gsub("\ ", "_", legend_d), ".pdf"), map, 
         path=share_plot_dir, device="pdf",
         dpi=300, width=15, height=12, unit="in", limitsize = FALSE)
}

######## Remove max and min in the hope that there is only one outlier

Abs_plot_dir <- paste0(plot_dir, "abs_change_map_NoOutlier/")
if (dir.exists(Abs_plot_dir) == F) {dir.create(path = Abs_plot_dir, recursive = T)}
share_plot_dir <- paste0(plot_dir, "share_change_map_NoOutlier/")
if (dir.exists(share_plot_dir) == F) {dir.create(path = share_plot_dir, recursive = T)}

for (a_col in colnames(inventory_AbsChange)[4:length(colnames(inventory_AbsChange))]){
  if (str_detect(a_col, pat="change")){
    s_year = substr(a_col, start = 11, stop = 14)
    e_year = substr(a_col, start = 17, stop = 20)
    if (str_detect(a_col, pat="asPerc")){
       legend_d <- paste0("Percentage change ", s_year, "_", e_year)
      } else {
       legend_d <- paste0("Absolute change ", s_year, "_", e_year)
      }
   } else {
       yr = tail(strsplit(a_col, split = "_")[[1]], 1)
       legend_d <- paste0("Invenory ", yr)

   }

  inventory_AbsChange_copy = copy(inventory_AbsChange)
  inventory_AbsChange_copy <- subset(inventory_AbsChange_copy, select = c("fips", a_col))
  inventory_AbsChange_copy <- inventory_AbsChange_copy %>% drop_na()

  minn = min(inventory_AbsChange_copy[, 2])
  maxx = max(inventory_AbsChange_copy[, 2])
  inventory_AbsChange_copy <- inventory_AbsChange_copy[inventory_AbsChange_copy[a_col] > minn ,]
  inventory_AbsChange_copy <- inventory_AbsChange_copy[inventory_AbsChange_copy[a_col] < maxx ,]
  
  
  map <- usmap::plot_usmap(data = inventory_AbsChange_copy, 
                           values = a_col,
                           labels = TRUE, label_color = "orange",
                           ) + 
         geom_polygon(data=states[[1]], aes(x=x, y=y, group=group), 
                      color = "yellow", fill = alpha(0.01), linewidth = 0.5) + 
         scale_fill_gradient2(low = "red", mid = "white", high = "blue", midpoint = 0, 
                              space = "Lab", na.value = "grey50", guide = "colourbar", aesthetics = "fill",
                              name=legend_d) +
         scale_colour_continuous_diverging() + 
         the_theme + 
         ggtitle(legend_d)
      
  map$layers[[2]]$aes_params$size <- 8

  ggsave(paste0(gsub("\ ", "_", legend_d), "_noOutlier.pdf"), map, 
         path=Abs_plot_dir, device="pdf",
         dpi=300, width=15, height=12, unit="in", limitsize = FALSE)
}



##### Changes as in National Share

for (a_col in colnames(inv_PercChangeShare)[4:length(colnames(inv_PercChangeShare))]){
  if (str_detect(a_col, pat="change")){
    s_year = unlist(strsplit(a_col, split = "_"))[2]
    e_year = unlist(strsplit(a_col, split = "_"))[3]
    legend_d <- paste0("Change of share ", s_year, "_", e_year)
   } else {
       yr = unlist(strsplit(a_col, split = "_"))[2]
       legend_d <- paste0("Invenory ", yr, " as percentage share")
   }

  inv_PercChangeShare_copy = copy(inv_PercChangeShare)
  inv_PercChangeShare_copy <- subset(inv_PercChangeShare_copy, select = c("fips", a_col))
  inv_PercChangeShare_copy <- inv_PercChangeShare_copy %>% drop_na()

  minn = min(inv_PercChangeShare_copy[, 2])
  maxx = max(inv_PercChangeShare_copy[, 2])
  inv_PercChangeShare_copy <- inv_PercChangeShare_copy[inv_PercChangeShare_copy[a_col] > minn ,]
  inv_PercChangeShare_copy <- inv_PercChangeShare_copy[inv_PercChangeShare_copy[a_col] < maxx ,]
  
  map <- usmap::plot_usmap(data = inv_PercChangeShare_copy, 
                           values = a_col,
                           labels = TRUE, label_color = "orange",
                           ) + 
         geom_polygon(data=states[[1]], aes(x=x, y=y, group=group), 
                      color = "yellow", fill = alpha(0.01), linewidth = 0.5) + 
         scale_fill_gradient2(low = "red", mid = "white", high = "blue", midpoint = 0, 
                              space = "Lab", na.value = "grey50", guide = "colourbar", aesthetics = "fill",
                              name=legend_d) +
         scale_colour_continuous_diverging() + 
         the_theme + 
         ggtitle(legend_d)
      
  map$layers[[2]]$aes_params$size <- 8

  ggsave(paste0(gsub("\ ", "_", legend_d), "_noOutlier.pdf"), map, 
         path=share_plot_dir, device="pdf",
         dpi=300, width=15, height=12, unit="in", limitsize = FALSE)
}



