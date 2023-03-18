library("zoo")
library("lubridate")
library("cowplot")

source("functions/probability_plots.R")
source("functions/combine_models.R")
source("functions/wsu_colors.R")

#### SET DATA ####

# df <- combine_models("data_48.46875_-122.40625", climate_proj = "rcp85") %>% filter(precip >=0) # DRY
df <- combine_models("data_48.71875_-121.09375", climate_proj = "rcp85") %>% 
      filter(precip >=0) # WET

# FOR Multiple climate projections
df <- map2("data_48.46875_-122.40625", c("rcp45", "rcp85"), combine_models) %>% 
      bind_rows() %>% 
      mutate(climate_proj = as.factor(climate_proj)) %>% 
      filter(precip >=0) # DRY

df <- map2("data_48.46875_-121.09375", c("rcp45", "rcp85"), combine_models) %>% 
  bind_rows() %>% 
  mutate(climate_proj = as.factor(climate_proj)) %>% 
  filter(precip >=0) # WET

# Plot Size
size <- 16

#### DAILY PROB ####
df_monthly_exceedance <- df %>% filter(group == 'hist') %>% 
  group_by(month) %>%
  mutate(prob = rank(precip)/(n()+1)) %>%
  summarise(precip_80 = nth(precip, which.min(abs(prob-0.8))),
            precip_90 = nth(precip, which.min(abs(prob-0.9))),
            precip_95 = nth(precip, which.min(abs(prob-0.95))))

df_monthly_prob <- full_join(df, df_monthly_exceedance, by = c("month")) %>% 
  filter(!is.na(group))

p_monthly_exceedance_val <- df_monthly_prob %>% 
  filter(group != "hist", month >= 10 | month <= 3) %>% 
  group_by(group, month, model) %>%
  mutate(prob = rank(-precip)/(n()+1)) %>%
  summarise(prob_80 = nth(prob, which.min(abs(precip-precip_80))),
            prob_90 = nth(prob, which.min(abs(precip-precip_90))),
            prob_95 = nth(prob, which.min(abs(precip-precip_95))),
            month_name = factor(first(month(time_stamp, label = TRUE)), 
                                levels = c('Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar'))) %>%
  gather(exceedance, prob, -month_name, -group, -month, -model) %>% 
  group_by(group, month, exceedance) %>% 
  mutate(prob_median = median(prob),
         hist_prob = case_when(exceedance == "prob_80" ~ 0.20,
                               exceedance == "prob_90" ~ 0.10,
                               exceedance == "prob_95" ~ 0.05)) %>% 
  plot_monthly_prob("Daily Prob")

df_octmar_exceedance <- df %>% filter(group == 'hist', month >= 10 | month <= 3) %>%
  mutate(prob = rank(precip)/(n()+1)) %>%
  summarise(precip_80 = nth(precip, which.min(abs(prob-0.8))),
            precip_90 = nth(precip, which.min(abs(prob-0.9))),
            precip_95 = nth(precip, which.min(abs(prob-0.95))))

p_octmar_exceedance_val <- df %>%
  filter(month >= 10 | month <= 3, group != "hist") %>% 
  group_by(group, model) %>%
  mutate(prob = rank(-precip)/(n()+1)) %>%
  summarise(prob_80 = nth(prob, which.min(abs(precip-df_octmar_exceedance$precip_80))),
            prob_90 = nth(prob, which.min(abs(precip-df_octmar_exceedance$precip_90))),
            prob_95 = nth(prob, which.min(abs(precip-df_octmar_exceedance$precip_95)))) %>% 
  gather(exceedance, prob, -model, -group) %>% 
  group_by(group, exceedance) %>% 
  mutate(prob_median = median(prob),
         hist_prob = case_when(exceedance == "prob_80" ~ 0.20,
                               exceedance == "prob_90" ~ 0.10,
                               exceedance == "prob_95" ~ 0.05)) %>% 
  plot_octmar_prob()

p_prob_daily <- plot_grid(p_monthly_exceedance_val,p_octmar_exceedance_val, nrow = 1, align = "vh", rel_widths = c(3.25, 1), axis = 'b')

# save_plot("figures/daily_exceedance_probability_dry.pdf", p_prob, base_aspect_ratio = 2, scale = 1, limitsize = FALSE)

#### 7 DAY PROB ####

df_7 <- df %>% group_by(group, model) %>% mutate(precip = rollsum(precip, 7, align = "right", fill = NA))

df_monthly_exceedance <- df_7 %>% filter(group == 'hist') %>% 
  group_by(month) %>%
  mutate(prob = rank(precip)/(n()+1)) %>%
  summarise(precip_80 = nth(precip, which.min(abs(prob-0.8))),
            precip_90 = nth(precip, which.min(abs(prob-0.9))),
            precip_95 = nth(precip, which.min(abs(prob-0.95))))

df_monthly_prob <- full_join(df_7, df_monthly_exceedance, by = c("month")) %>% 
  filter(!is.na(group))

p_monthly_exceedance_val <- df_monthly_prob %>% 
  filter(group != "hist", month >= 10 | month <= 3) %>% 
  group_by(group, month, model) %>%
  mutate(prob = rank(-precip)/(n()+1)) %>%
  summarise(prob_80 = nth(prob, which.min(abs(precip-precip_80))),
            prob_90 = nth(prob, which.min(abs(precip-precip_90))),
            prob_95 = nth(prob, which.min(abs(precip-precip_95))),
            month_name = factor(first(month(time_stamp, label = TRUE)), 
                                levels = c('Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar'))) %>%
  gather(exceedance, prob, -month_name, -group, -month, -model) %>% 
  group_by(group, month, exceedance) %>% 
  mutate(prob_median = median(prob),
         hist_prob = case_when(exceedance == "prob_80" ~ 0.20,
                               exceedance == "prob_90" ~ 0.10,
                               exceedance == "prob_95" ~ 0.05)) %>% 
  plot_monthly_prob("7 Day Prob")

df_octmar_exceedance <- df_7 %>% filter(group == 'hist', month >= 10 | month <= 3) %>%
  mutate(prob = rank(precip)/(n()+1)) %>%
  summarise(precip_80 = nth(precip, which.min(abs(prob-0.8))),
            precip_90 = nth(precip, which.min(abs(prob-0.9))),
            precip_95 = nth(precip, which.min(abs(prob-0.95))))

p_octmar_exceedance_val <- df_7 %>%
  filter(month >= 10 | month <= 3, group != "hist") %>% 
  group_by(group, model) %>%
  mutate(prob = rank(-precip)/(n()+1)) %>%
  summarise(prob_80 = nth(prob, which.min(abs(precip-df_octmar_exceedance$precip_80))),
            prob_90 = nth(prob, which.min(abs(precip-df_octmar_exceedance$precip_90))),
            prob_95 = nth(prob, which.min(abs(precip-df_octmar_exceedance$precip_95)))) %>% 
  gather(exceedance, prob, -model, -group) %>% 
  group_by(group, exceedance) %>% 
  mutate(prob_median = median(prob),
         hist_prob = case_when(exceedance == "prob_80" ~ 0.20,
                               exceedance == "prob_90" ~ 0.10,
                               exceedance == "prob_95" ~ 0.05)) %>% 
  plot_octmar_prob()

p_prob_7day <- plot_grid(p_monthly_exceedance_val,p_octmar_exceedance_val, nrow = 1, align = "vh", rel_widths = c(3.25, 1), axis = 'b')


# save_plot("figures/7_day_exceedance_probability_dry.pdf", p_prob, base_aspect_ratio = 2, scale = 1, limitsize = FALSE)

#### MONTH PROB ####

df_monthly_exceedance <- df %>% filter(group == 'hist') %>% 
  group_by(month, year) %>%
  summarise(precip = sum(precip)) %>% 
  group_by(month) %>%
  mutate(prob = rank(precip)/(n()+1)) %>%
  summarise(precip_80 = nth(precip, which.min(abs(prob-0.8))),
            precip_90 = nth(precip, which.min(abs(prob-0.9))),
            precip_95 = nth(precip, which.min(abs(prob-0.95))))

df_monthly_prob <- full_join(df, df_monthly_exceedance, by = c("month")) %>% 
  filter(!is.na(group))

p_monthly_exceedance_val <- df_monthly_prob %>% 
  filter(group != "hist", month >= 10 | month <= 3) %>% 
  group_by(group, month, model, year) %>%
  summarise(precip = sum(precip),
            precip_80 = first(precip_80),
            precip_90 = first(precip_90),
            precip_95 = first(precip_95),
            time_stamp = first(time_stamp)) %>% 
  group_by(group, month, model) %>% 
  mutate(prob = rank(-precip)/(n()+1)) %>%
  summarise(prob_80 = nth(prob, which.min(abs(precip-precip_80))),
            prob_90 = nth(prob, which.min(abs(precip-precip_90))),
            prob_95 = nth(prob, which.min(abs(precip-precip_95))),
            month_name = factor(first(month(time_stamp, label = TRUE)), 
                                levels = c('Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar'))) %>%
  gather(exceedance, prob, -month_name, -model, -month, -group) %>% 
  group_by(group, month, exceedance) %>% 
  mutate(prob_median = median(prob),
         hist_prob = case_when(exceedance == "prob_80" ~ 0.20,
                               exceedance == "prob_90" ~ 0.10,
                               exceedance == "prob_95" ~ 0.05)) %>% 
  plot_monthly_prob("Monthly Prob")

df_octmar_exceedance <- df %>% filter(group == 'hist', month >= 10 | month <= 3) %>%
  group_by(group, month, model, year) %>% 
  summarise(precip = sum(precip)) %>% 
  mutate(prob = rank(precip)/(n()+1)) %>%
  summarise(precip_80 = nth(precip, which.min(abs(prob-0.8))),
            precip_90 = nth(precip, which.min(abs(prob-0.9))),
            precip_95 = nth(precip, which.min(abs(prob-0.95))))

p_octmar_exceedance_val <- df %>%
  filter(month >= 10 | month <= 3, group != "hist") %>% 
  group_by(group, month, model, year) %>% 
  summarise(precip = sum(precip)) %>% 
  group_by(group, model) %>%
  mutate(prob = rank(-precip)/(n()+1)) %>%
  summarise(prob_80 = nth(prob, which.min(abs(precip-df_octmar_exceedance$precip_80))),
            prob_90 = nth(prob, which.min(abs(precip-df_octmar_exceedance$precip_90))),
            prob_95 = nth(prob, which.min(abs(precip-df_octmar_exceedance$precip_95)))) %>%  
  gather(exceedance, prob, -model, -group) %>% 
  group_by(group, exceedance) %>% 
  mutate(prob_median = median(prob),
         hist_prob = case_when(exceedance == "prob_80" ~ 0.20,
                               exceedance == "prob_90" ~ 0.10,
                               exceedance == "prob_95" ~ 0.05)) %>% 
  plot_octmar_prob()

p_prob_monthly <- plot_grid(p_monthly_exceedance_val,p_octmar_exceedance_val, nrow = 1, align = "vh", rel_widths = c(3.25, 1), axis = 'b')

p_all <- plot_grid(p_prob_daily, p_prob_7day, p_prob_monthly, ncol = 1, align = "vh")

# save_plot("figures/monthly_exceedance_probability_wet.pdf", p_prob, base_aspect_ratio = 2, scale = 1, limitsize = FALSE)

save_plot("figures/exceedance_probability_dry.pdf", p_all, base_aspect_ratio = .5, scale = 3.5, limitsize = FALSE)

#### OTHER ####

# df %>% group_by(model) %>%
#   filter(precip > 0, month >= 10 | month <= 3) %>%
#   group_by(water_year, model) %>%
#   summarise(precip = mean(precip)) %>%
#   mutate(prob = rank(-precip)/(n()+1)) %>%
#   ggplot() +
#   geom_line(aes(x = precip, y = prob, color = model)) +
#   scale_x_log10() +
#   scale_y_log10() +
#   labs(x = "Precipitation (mm)", y = "Probability of Exceedance") +
#   theme(legend.title = element_blank(),
#         legend.position = c(.1, .2))
# 
# df_prob <- df %>% group_by(model, group) %>%
#   filter(precip > 0, !is.na(group)) %>% 
#   mutate(prob = rank(-precip)/(n()+1))
# 
# 
# ggplot() +
#   geom_smooth(data = filter(df_prob, group != 'hist'), aes(x = precip, y = prob, color = group, fill = group)) +
#   # geom_line(data = filter(df_prob, group == 'hist'), aes(x = precip, y = prob), color = "black", size = 1.2) +
#   scale_x_log10() +
#   scale_y_log10(breaks = c(0.0001, 0.001, 0.001, 0.01, 0.1, 1)) +
#   labs(x = "Precipitation (mm)", y = "Probability of Exceedance") +
#   theme(legend.title = element_blank(),
#         legend.position = c(.1, .25)) +
#   scale_color_wsu(palette = 'rev') +
#   scale_fill_wsu(palette = 'rev') 
# 
# save_plot("figures/exceedance_wet.pdf", p, base_aspect_ratio = 1.618)
# 
df_cum_precip <- df %>% mutate(water_date = time_stamp %m+% months(3),
              water_week = week(water_date),
              water_year = year(water_date),
              climate_proj = ifelse(is.na(climate_proj), "historical", as.character(climate_proj))) %>%
  filter(precip > 0, month >= 10 | month <= 3) %>%
  group_by(model, climate_proj, water_year, water_week, group) %>%
  summarise(precip = sum(precip), time_stamp = first(time_stamp)) %>%
  group_by(model, climate_proj, water_year, group) %>%
  mutate(cum_precip = cumsum(precip)) %>%
  group_by(climate_proj, water_week, group) %>%
  summarise(time_stamp = first(time_stamp),
            model = first(model),
            cum_precip_median = median(cum_precip),
            cum_precip_low = quantile(cum_precip, .25),
            cum_precip_high = quantile(cum_precip, .75)) %>%
  mutate(month_dec = month(time_stamp) + (day(time_stamp) - 1)/days_in_month(time_stamp),
         water_date = time_stamp %m+% months(3),
         CDate = as.Date(paste0(ifelse(month(time_stamp) < 10, "1901", "1900"),
                              "-", month(time_stamp), "-", day(time_stamp))))

p <- df_cum_precip %>% filter(!is.na(group)) %>% 
     ggplot() +
     geom_line(aes(x = CDate, y = cum_precip_median, color = climate_proj, linetype = group)) +
     geom_ribbon(aes(x = CDate, ymin = cum_precip_low, ymax = cum_precip_high, fill = climate_proj), alpha = 0.25) +
     labs(y = "Mean Cumalitive Weekly Precip (mm)") +
     theme(legend.title = element_blank(),
           axis.title.x = element_blank(),
           legend.position = c(.75, 0.15)) +
     scale_x_date(date_labels = "%b", date_breaks = "1 month") +
     scale_color_wsu() +
     scale_fill_wsu() 

save_plot("figures/cum_precip_wet.pdf", p, base_aspect_ratio = 1.618)


# df_cum_precip <- df %>% mutate(water_date = time_stamp %m+% months(3),
#                                water_week = week(water_date),
#                                water_year = year(water_date)) %>% 
#   filter(precip > 0, month >= 10 | month <= 3) %>% 
#   group_by(model, climate_proj, group, water_year, water_week) %>% 
#   summarise(precip = sum(precip), time_stamp = first(time_stamp)) %>% 
#   group_by(model, climate_proj, group, water_year) %>% 
#   mutate(cum_precip = cumsum(precip)) %>% 
#   group_by(model, climate_proj, group, water_week) %>% 
#   summarise(cum_precip = mean(cum_precip), time_stamp = first(time_stamp),
#             month_dec = month(time_stamp) + (day(time_stamp) - 1)/days_in_month(time_stamp),
#             CDate=as.Date(paste0(ifelse(month(time_stamp) < 10, "1901", "1900"),
#                               "-", month(time_stamp), "-", day(time_stamp)))) %>% 
#   group_by(climate_proj, group, CDate) %>% 
#   summarise(mean_precip = mean(cum_precip),
#             sd_precip = sd(cum_precip))
# 
# 
# 
# p <- ggplot() +
#   geom_line(data = filter(df_cum_precip, !is.na(group)),
#             aes(x = CDate, y = mean_precip, color = group, group = interaction(climate_proj, group))) +
#   geom_ribbon(data = filter(df_cum_precip, !is.na(group)),
#             aes(x = CDate, ymax = mean_precip + sd_precip, ymin = mean_precip - sd_precip, fill = group, group = interaction(climate_proj, group)), alpha = 0.5) +
#   labs(x = " ", y = "Mean Cumalitive Weekly Precip (mm)") +
#   theme(legend.title = element_blank(),
#         legend.position = c(.75, 0.15)) +
#   # scale_y_continuous(breaks = seq(0,1250, 250)) +
#   scale_x_date(date_labels = "%b", date_breaks = "1 month") +
#   scale_color_wsu(palette = "rev") +
#   scale_fill_wsu(palette = "rev")
# 
# save_plot("figures/cum_precip_wet.pdf", p, base_aspect_ratio = 1.618)
# 
# 
# 
# 

### DRY DAYS ####
p_monthly_dry <- df %>% group_by(month, year, group, model, climate_proj) %>%
  filter(precip == 0, month >= 10 | month <= 3, !is.na(group)) %>%
  summarise(dry_days = n(),
            month_name = factor(first(month(time_stamp, label = TRUE)),
                                levels = c('Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar'))) %>%
  ggplot() +
  geom_boxplot(aes(x = group, y = dry_days, color = group, fill = group), alpha = 0.75) +
  # geom_jitter(aes(x = group, y = total_precip, color = group), alpha = 0.5, width = 0.05) +
  theme_classic(base_size = size) +
  facet_grid(.~month_name, switch = "x") +
  theme(axis.text.x = element_blank(),
        axis.ticks.x = element_blank(),
        axis.title.x = element_blank(),
        legend.title = element_blank(),
        legend.position = "top",
        strip.background = element_blank(),
        strip.placement = "outside") +
  scale_color_wsu() +
  scale_fill_wsu() +
  labs(y = "# Dry Days (Dry)") +
  ylim(0, 31)

# p <- plot_grid(p_monthly_dry, p_monthly_dry_model, ncol = 1, rel_heights = c(1, 5.5))

# save_plot("figures/dry_days_models.pdf", p, base_aspect_ratio = 1.1, scale=2.75)

p_octmar_dry <- df %>% mutate(water_year = year(time_stamp + month(3))) %>%
                group_by(group, water_year, model, climate_proj) %>%
                filter(month >= 10 | month <= 3, precip == 0, !is.na(group)) %>%
                summarise(dry_days = n()) %>%
                ggplot() +
                geom_boxplot(aes(x = group, y = dry_days, color = group, fill = group), alpha = 0.75) +
                # geom_jitter(aes(x = group, y = total_precip, color = group), alpha = 0.5, width = 0.05) +
                theme_classic(base_size = size) +
                # facet_grid(.~month_name, switch = "x") +
                theme(axis.text.x = element_blank(),
                      axis.ticks.x = element_blank(),
                      axis.title.y = element_blank(),
                      legend.title = element_blank(),
                      legend.position = "none") +
                scale_color_wsu() +
                scale_fill_wsu() +
                labs(x = "Oct - Mar") +
                ylim(0, 110)

p_dry_dry <- plot_grid(p_monthly_dry, p_octmar_dry, rel_widths = c(5.5, 1), align = 'vh', nrow = 1, axis = "b")


p_dry <- plot_grid(p_dry_dry, p_dry_wet, align = 'vh', ncol = 1)

save_plot("figures/dry_days.pdf", p_dry, base_aspect_ratio = 1.5, scale = 2, limitsize = FALSE)

#### Histogram of precip ####

p <- ggplot(filter(df, !is.na(group))) +
     geom_density(aes(precip, color = group), alpha = 0.1) +
     ylim(0, 0.10) +
     xlim(0, 25) +
     scale_fill_wsu() +
     scale_color_wsu() +
     # facet_grid(group~.) +
     theme_classic() +
     theme(legend.position = "bottom")

save_plot("figures/precip_density_dry_zoom.pdf", 
          p, 
          base_aspect_ratio = 1.618, 
          scale = 2)





