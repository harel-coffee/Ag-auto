library(tidyverse)
library(lubridate)
library(stringr)
library(cowplot)
library(rgdal)
library(sf)

# read county JSON file ####
skagit <- readOGR("geo/Skagit.geo.json") # Skagit County 
snohomish <- readOGR("geo/Snohomish.geo.json") # Snohomish County
whatcom <- readOGR("geo/Whatcom.geo.json") # Whatcom County

# Bind all counties data #
counties <- rbind(skagit, snohomish, whatcom, makeUniqueIDs = TRUE)
counties <- st_as_sf(counties)

# read data ####
df <- read_rds("data/precip_summary.rds") 
# select rcp 8.5
df <- df %>% filter(climate_proj %in% c("rcp85", NA))

# Plot diff function
plot_diff_contour <- function(df, time_scale){
  
  ggplot(df) +
    geom_raster(aes(x = lng, y = lat, fill = precip),
                alpha = .5, interpolate = FALSE) +
    geom_contour(aes(x = lng, y = lat, z = precip), bins = 6) +
    geom_sf(data = counties, fill = NA, lwd = 1.5) +
    scale_fill_gradient2(low = "blue", high = "red", labels = scales::percent) +
    facet_grid(. ~ group) +
    labs(fill = "Percent Change", 
         title = paste("Difference from Historical Median", time_scale, "Precip")) +
    NULL
  
}

# yearly difference ####
med_yearly <- df %>% 
  group_by(lat, lng, group, model, year) %>% 
  summarise(precip = sum(precip)) %>% 
  group_by(lat, lng, group) %>% 
  summarise(precip = median(precip))

med_hist_yearly <- med_yearly %>% 
  filter(group == "hist") %>% 
  select(lat, lng, hist_precip = precip)

p_yearly <- med_yearly %>% 
  full_join(med_hist_yearly) %>% 
  mutate(precip = (precip - hist_precip)/hist_precip) %>% 
  plot_diff_contour("Yearly")

# oct-mar
med_octmar <- df %>% 
  filter(!between(month, 4, 9)) %>%
  mutate(date_time = ymd(paste(year, month, "01", sep = "-")),
         water_year = year(date_time %m+% months(3))) %>%
  group_by(lat, lng, group, model, water_year) %>% 
  summarise(precip = sum(precip)) %>% 
  group_by(lat, lng, group) %>% 
  summarise(precip = median(precip))

med_hist_octmar <- med_octmar %>% 
  ungroup() %>% 
  filter(group == "hist") %>% 
  select(lat, lng, hist_precip = precip)

p_octmar <- med_octmar %>% 
  full_join(med_hist_octmar) %>% 
  mutate(precip = (precip - hist_precip)/hist_precip) %>% 
  plot_diff_contour("Oct-Mar")

#seasonal ####

df_season <- df %>%
  mutate(season = case_when(month <= 3 ~ "Jan-Mar",
                            between(month, 4, 6) ~ "Apr-Jun",
                            between(month, 7, 9) ~ "Jul-Sep",
                            month >= 10 ~ "Oct-Dec")) %>% 
  group_by(lat, lng, year, model, group, season) %>%
  summarise(precip = sum(precip)) %>%
  group_by(lat, lng, group, season) %>%
  summarise(precip = median(precip))

df_season_hist <- df_season %>% filter(group == "hist") %>% 
  ungroup() %>% 
  select(lat, lng, season, hist_precip = precip) %>% 
  full_join(df_season) %>% 
  mutate(precip = (precip - hist_precip)/hist_precip)

plot_seasonal_contours <- function(df, sel_season){
  df %>% filter(season == sel_season) %>% 
    ggplot() +
    geom_raster(aes(x = lng, y = lat, fill = precip),
                alpha = .5,
                interpolate = FALSE) +
    geom_contour(aes(x = lng, y = lat, z = precip), bins = 6) +
    geom_sf(data = counties, fill = NA, lwd = 1.5) +
    scale_fill_gradient2(low = "blue", high = "red", labels = scales::percent) +
    facet_grid(. ~ group) +
    labs(fill = "Percent Changes", 
         title = paste("Difference from Historical Median", sel_season, "Precip")) +
    # theme(legend.position = "bottom") +
    # labs(title = select_group) +
    NULL
}

# combine plots ####
df_plots <- tibble(data = list(df_season_hist),
                   seasons = as_factor(unique(df_season$season), "Jan-Mar")) %>% 
  mutate(plot = map2(data, seasons, plot_seasonal_contours))

# p_seasons <- do.call(plot_grid, c(df_plots$plot, ncol = 1))

p <- plot_grid(df_plots$plot[[1]], 
               df_plots$plot[[2]], 
               df_plots$plot[[3]], 
               df_plots$plot[[4]], 
               p_octmar, p_yearly, ncol = 1)


save_plot("figures/diff_hist_contour.pdf", p, base_aspect_ratio = 0.8, scale = 4)
