library(tidyverse)
library(lubridate)
library(stringr)
library(cowplot)
library(rgdal)
library(sf)

# read JSON ####
skagit <- readOGR("geo/Skagit.geo.json") # Skagit County
snohomish <- readOGR("geo/Snohomish.geo.json") # Snohomish County
whatcom <- readOGR("geo/Whatcom.geo.json") # Whatcom County

# Bind all counties data #
counties <- rbind(skagit, snohomish, whatcom, makeUniqueIDs = TRUE)
counties <- st_as_sf(counties)

# ggplot() +
#   geom_sf(data = counties, fill = NA, lwd = 1.5)


# Merge historical and models ####
# df_hist <- read_rds("data/precip_summary_hist.rds") %>%
#   mutate(climate_proj = NA,
#          model = "historical",
#          group = "hist") %>%
#   select(-file_name) %>%
#   as_tibble()
# 
# df_all <- read_rds("data/precip_summary_all.rds") %>%
#   filter(model != "historical") %>%
#   mutate(coord = str_sub(file_name, 37,-5)) %>%
#   separate(coord, c("lat", "lng"), "_", convert = TRUE) %>%
#   select(-file_name)
# 
# df <- bind_rows(df_hist, df_all) %>%
#   mutate(precip = precip/25.4,
#          group = as_factor(group, "hist"))
# 
# write_rds(df, "data/precip_summary.rds")

df <- read_rds("data/precip_summary.rds")


# Contour Model plot ####
plot_contour_model <- function(df, select_group) {
  df %>%
    filter(group %in% c(select_group, "hist"),
           climate_proj %in% c("rcp45", NA)) %>%
    group_by(lat, lng, climate_proj, group, model, year) %>% 
    summarise(precip = sum(precip)) %>% 
    group_by(lat, lng, climate_proj, group, model) %>% 
    summarise(precip = median(precip)) %>% 
    ggplot() +
    geom_raster(aes(x = lng, y = lat, fill = precip),
                alpha = .5,
                interpolate = FALSE) +
    geom_contour(aes(x = lng, y = lat, z = precip), bins = 6) +
    geom_sf(data = counties, fill = NA, lwd = 1.5) +
    scale_fill_viridis_c() +
    # facet_grid(model~month) +
    facet_wrap(vars(model), ncol = 2) +
    labs(fill = "mean yearly precip") +
    theme(legend.position = "bottom") +
    labs(title = select_group) +
    NULL
}

p_2040 <- plot_contour_model(df, "2040s")
p_2060 <- plot_contour_model(df, "2060s")
p_2080 <- plot_contour_model(df, "2080s")

p <- plot_grid(p_2040, p_2060, p_2080, ncol = 3)
save_plot("yearly_contours_rcp45.pdf",
          p,
          scale = 4,
          base_aspect_ratio = 1.2)

#  October-March
df_oct_mar <- bind_rows(df_hist, df_all) %>%
  filter(climate_proj %in% c("rcp85", NA)) %>%
  mutate(date_time = ymd(paste(year, month, "01", sep = "-")),
         water_year = year(date_time %m+% months(3))) %>%
  filter(!between(month, 4, 9)) %>%
  group_by(lat, lng, water_year, model, group) %>%
  summarise(precip = sum(precip)) %>%
  group_by(lat, lng, group, model) %>%
  summarise(precip = mean(precip) / 25.4)

p_2040 <- plot_contour_model(df_oct_mar, "2040s")
p_2060 <- plot_contour_model(df_oct_mar, "2060s")
p_2080 <- plot_contour_model(df_oct_mar, "2080s")

p <- plot_grid(p_2040, p_2060, p_2080, ncol = 3)
save_plot("octmar_contours.pdf",
          p,
          scale = 4,
          base_aspect_ratio = 1.2)


# average over models
p_yearly <- df %>% 
  group_by(lat, lng, climate_proj, group, model, year) %>% 
  summarise(precip = sum(precip)) %>% 
  group_by(lat, lng, climate_proj, group) %>% 
  summarise(precip = median(precip)) %>% 
  filter(climate_proj %in% c("rcp45", NA)) %>%
  ggplot() +
  geom_raster(aes(x = lng, y = lat, fill = precip),
              alpha = .5,
              interpolate = FALSE) +
  geom_contour(aes(x = lng, y = lat, z = precip), bins = 6) +
  geom_sf(data = counties, fill = NA, lwd = 1.5) +
  scale_fill_viridis_c() +
  facet_grid(. ~ group) +
  labs(fill = "median yearly precip") +
  # labs(title = select_group) +
  NULL

df_oct_mar <- df %>%
  mutate(date_time = ymd(paste(year, month, "01", sep = "-")),
         water_year = year(date_time %m+% months(3))) %>%
  filter(!between(month, 4, 9),
         climate_proj %in% c("rcp45", NA)) %>%
  group_by(lat, lng, water_year, model, group) %>%
  summarise(precip = sum(precip)) %>%
  group_by(lat, lng, group) %>%
  summarise(precip = median(precip))

p_octmar <- df_oct_mar %>%
  ggplot() +
  geom_raster(aes(x = lng, y = lat, fill = precip),
              alpha = .5,
              interpolate = FALSE) +
  geom_contour(aes(x = lng, y = lat, z = precip), bins = 6) +
  geom_sf(data = counties, fill = NA, lwd = 1.5) +
  scale_fill_viridis_c() +
  facet_grid(. ~ group) +
  labs(fill = "mean oct-mar precip") +
  # theme(legend.position = "bottom") +
  # labs(title = select_group) +
  NULL

save_plot("octmar_contours.pdf", p, scale = 2, base_aspect_ratio = 2)


df_season <- df %>%
  filter(climate_proj %in% c("rcp45", NA)) %>%
  mutate(season = case_when(month <= 3 ~ "Jan-Mar",
                            between(month, 4, 6) ~ "Apr-Jun",
                            between(month, 7, 9) ~ "Jul-Sep",
                            month >= 10 ~ "Oct-Dec")) %>% 
  # mutate(date_time = ymd(paste(year, month, "01", sep = "-")),
  #        water_year = year(date_time %m+% months(3))) %>%
  # filter(between(month, 1, 3)) %>%
  group_by(lat, lng, year, model, group, season) %>%
  summarise(precip = sum(precip)) %>%
  group_by(lat, lng, group, season) %>%
  summarise(precip = mean(precip))

plot_seasonal_contours <- function(df, sel_season){
  df %>% filter(season == sel_season) %>% 
    ggplot() +
    geom_raster(aes(x = lng, y = lat, fill = precip),
                alpha = .5,
                interpolate = FALSE) +
    geom_contour(aes(x = lng, y = lat, z = precip), bins = 6) +
    geom_sf(data = counties, fill = NA, lwd = 1.5) +
    scale_fill_viridis_c() +
    facet_grid(. ~ group) +
    labs(fill = "mean seasonal precip (in)", title = sel_season) +
    # theme(legend.position = "bottom") +
    # labs(title = select_group) +
    NULL
}

p_win <- plot_seasonal_contours(df_season, "Jan-Mar") 
p_spr <- plot_seasonal_contours(df_season, "Apr-Jun") 
p_sum <- plot_seasonal_contours(df_season, "Jul-Sep") 
p_fal <- plot_seasonal_contours(df_season, "Oct-Dec") 


p <- plot_grid(p_win, p_spr, p_sum, p_fal, p_octmar, p_yearly, ncol = 1)

save_plot("season_contours_rcp85.pdf", p, scale = 5, base_aspect_ratio = .8)


# Difference from historical ####
med_yearly <- df %>% 
  group_by(lat, lng, climate_proj, group, model, year) %>% 
  summarise(precip = sum(precip)) %>% 
  group_by(lat, lng, climate_proj, group) %>% 
  summarise(precip = median(precip))

med_hist <- med_yearly %>% 
  ungroup() %>% 
  filter(group == "hist") %>% 
  select(lat, lng, hist_precip = precip)

p <- med_yearly %>% 
  filter(group != "hist") %>% 
  full_join(med_hist) %>% 
  mutate(precip = precip - hist_precip) %>% 
  ggplot() +
  geom_raster(aes(x = lng, y = lat, fill = precip),
              alpha = .5, interpolate = FALSE) +
  geom_contour(aes(x = lng, y = lat, z = precip), bins = 6) +
  geom_sf(data = counties, fill = NA, lwd = 1.5) +
  scale_fill_viridis_c() +
  facet_grid(climate_proj ~ group) +
  labs(fill = "difference from historical median yearly precip") +
  theme(legend.position = "bottom") +
  # labs(title = select_group) +
  NULL

save_plot("diff_hist_contour.pdf", p, base_aspect_ratio = 2, scale = 2)

p <- med_yearly %>% 
  filter(group != "hist") %>% 
  full_join(med_hist) %>% 
  mutate(precip = precip - hist_precip) %>% 
  ggplot() +
  geom_boxplot(aes(x = climate_proj, y = precip, fill = group)) +
  labs(y = "difference from historical precip")

save_plot("precip_diff.pdf", p, scale = 1.2, base_aspect_ratio = 1.618)