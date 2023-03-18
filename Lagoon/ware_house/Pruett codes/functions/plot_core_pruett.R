

runoff_plots <- function(df){
  
  size <- 16
  
  df <- df %>% filter(!is.na(group)) %>% 
    mutate(combined = runoff+baseflow, 
           water_year = year(time_stamp + month(3)))
    
    # set plot scale
    size <- 14
    
    #### PROBABILITY PLOT MONTHLY ####
    
    # Calculate monthly exceedance
    df_monthly_exceedance <- df %>% filter(group == 'hist') %>% 
      group_by(month, year) %>%
      summarise(combined = sum(combined)) %>% 
      group_by(month) %>% 
      mutate(prob = rank(combined)/(n()+1)) %>%
      summarise(combined_80 = nth(combined, which.min(abs(prob-0.8))),
                combined_90 = nth(combined, which.min(abs(prob-0.9))),
                combined_95 = nth(combined, which.min(abs(prob-0.95))))
    
    df_monthly_prob <- full_join(df, df_monthly_exceedance, by = "month")
    
    p_monthly_exceedance_val <- df_monthly_prob %>% 
      group_by(group, month, year) %>% 
      summarise(combined = sum(combined),
                combined_80 = first(combined_80),
                combined_90 = first(combined_90),
                combined_95 = first(combined_95)) %>% 
      group_by(group, month) %>% 
      mutate(prob = rank(-combined)/(n()+1)) %>%
      summarise(prob_80 = nth(prob, which.min(abs(combined-combined_80))),
                prob_90 = nth(prob, which.min(abs(combined-combined_90))),
                prob_95 = nth(prob, which.min(abs(combined-combined_95)))) %>%
      ggplot() +
      geom_col(aes(x = group, y = prob_80, color = group, fill = group), alpha = 0.33) +
      geom_col(aes(x = group, y = prob_90, color = group, fill = group), alpha = 0.33) +
      geom_col(aes(x = group, y = prob_95, color = group, fill = group), alpha = 0.33) +
      facet_grid(.~month, switch = "x") +
      labs(x = " ", y = "Monthly Prob") +
      theme_classic(base_size = size) +
      theme(axis.text.x=element_blank(), 
            axis.ticks.x = element_blank(),
            legend.title = element_blank(), 
            legend.position = "none",
            strip.background = element_blank(),
            strip.text = element_blank()) +
      scale_color_wsu() +
      scale_fill_wsu()
    
    df_yearly_exceedance <- df %>% filter(group == 'hist') %>% 
      group_by(month, year) %>% 
      summarise(combined = sum(combined)) %>% 
      ungroup() %>% 
      mutate(prob = rank(combined)/(n()+1)) %>%
      summarise(combined_80 = nth(combined, which.min(abs(prob-0.8))),
                combined_90 = nth(combined, which.min(abs(prob-0.9))),
                combined_95 = nth(combined, which.min(abs(prob-0.95))))
    
    
    p_yearly_exceedance_val <- df %>% 
      group_by(month, group, year) %>% 
      summarise(combined = sum(combined)) %>% 
      group_by(group) %>% 
      mutate(prob = rank(-combined)/(n()+1)) %>%
      summarise(prob_80 = nth(prob, which.min(abs(combined-df_yearly_exceedance$combined_80))),
                prob_90 = nth(prob, which.min(abs(combined-df_yearly_exceedance$combined_90))),
                prob_95 = nth(prob, which.min(abs(combined-df_yearly_exceedance$combined_95)))) %>%
      ggplot() +
      # geom_col(aes(x = group, y = prob_80, color = group, fill = group), alpha = 0.33) +
      # geom_col(aes(x = group, y = prob_90, color = group, fill = group), alpha = 0.33) +
      # geom_col(aes(x = group, y = prob_95, color = group, fill = group), alpha = 0.33) +
      geom_col(aes(x = group, y = prob_80, color = group, fill = group), alpha = 0.33) +
      geom_col(aes(x = group, y = prob_90, color = group, fill = group), alpha = 0.33) +
      geom_col(aes(x = group, y = prob_95, color = group, fill = group), alpha = 0.33) +
      labs(x = " ", y = " ") +
      theme_classic(base_size = size) +
      theme(axis.text.x=element_blank(), 
            axis.ticks.x = element_blank(),
            legend.title = element_blank(), 
            legend.position = "none",
            strip.background = element_blank(),
            strip.text = element_blank()) +
      scale_color_wsu() +
      scale_fill_wsu()
    
    df_octmar_exceedance <- df %>% filter(group == 'hist', month >= 10 | month <= 3) %>%
      group_by(month, group, water_year) %>% 
      summarise(combined = sum(combined)) %>% 
      ungroup() %>% 
      mutate(prob = rank(combined)/(n()+1)) %>%
      summarise(combined_80 = nth(combined, which.min(abs(prob-0.8))),
                combined_90 = nth(combined, which.min(abs(prob-0.9))),
                combined_95 = nth(combined, which.min(abs(prob-0.95))))
    
    p_octmar_exceedance_val <- df %>% filter(month >= 10 | month <= 3) %>%
      group_by(month, group, year) %>% 
      summarise(combined = sum(combined)) %>% 
      group_by(group) %>% 
      mutate(prob = rank(-combined)/(n()+1)) %>%
      summarise(prob_80 = nth(prob, which.min(abs(combined-df_octmar_exceedance$combined_80))),
                prob_90 = nth(prob, which.min(abs(combined-df_octmar_exceedance$combined_90))),
                prob_95 = nth(prob, which.min(abs(combined-df_octmar_exceedance$combined_95)))) %>%
      ggplot() +
      geom_col(aes(x = group, y = prob_80, color = group, fill = group), alpha = 0.33) +
      geom_col(aes(x = group, y = prob_90, color = group, fill = group), alpha = 0.33) +
      geom_col(aes(x = group, y = prob_95, color = group, fill = group), alpha = 0.33) +
      labs(x = " ", y = " ") +
      theme_classic(base_size = size) +
      theme(axis.text.x=element_blank(),
            axis.ticks.x = element_blank(),
            legend.title = element_blank(),
            legend.position = "none",
            strip.background = element_blank(),
            strip.text = element_blank()) +
      scale_color_wsu() +
      scale_fill_wsu()
    
    p_prob_m <- plot_grid(p_monthly_exceedance_val, p_yearly_exceedance_val,p_octmar_exceedance_val, nrow = 1, align = "h", rel_widths = c(6, 1, 1), axis = 'b')
    

    
    #### EXCEEDANCE PLOT MONTHLY ####
    
    p_monthly_exceedance <- df %>% group_by(group, month, year) %>%
      summarise(combined = sum(combined)) %>% 
      group_by(group, month) %>% 
      mutate(prob = rank(combined)/(n()+1)) %>%
      summarise(combined_80 = nth(combined, which.min(abs(prob-0.8))),
                combined_90 = nth(combined, which.min(abs(prob-0.9))),
                combined_95 = nth(combined, which.min(abs(prob-0.95)))) %>%
      ggplot() +
      geom_col(aes(x = group, y = combined_80, color = group, fill = group), alpha = 0.33) +
      geom_col(aes(x = group, y = combined_90, color = group, fill = group), alpha = 0.33) +
      geom_col(aes(x = group, y = combined_95, color = group, fill = group), alpha = 0.33) +
      facet_grid(.~month, switch = "x") +
      labs(x = " ", y = "Month Exceed") +
      theme_classic(base_size = size) +
      theme(axis.text.x=element_blank(),
            axis.ticks.x = element_blank(),
            legend.title = element_blank(),
            legend.position = "none",
            strip.background = element_blank(),
            strip.text = element_blank()) +
      scale_color_wsu() +
      scale_fill_wsu()
    
    p_yearly_exceedance <- df %>% group_by(group, month, water_year) %>%
      summarise(combined = sum(combined)) %>% 
      group_by(group) %>% 
      mutate(prob = rank(combined)/(n()+1)) %>%
      summarise(combined_80 = nth(combined, which.min(abs(prob-0.8))),
                combined_90 = nth(combined, which.min(abs(prob-0.9))),
                combined_95 = nth(combined, which.min(abs(prob-0.95)))) %>%
      ggplot() +
      geom_col(aes(x = group, y = combined_80, color = group, fill = group), alpha = 0.33) +
      geom_col(aes(x = group, y = combined_90, color = group, fill = group), alpha = 0.33) +
      geom_col(aes(x = group, y = combined_95, color = group, fill = group), alpha = 0.33) +
      labs(x = " ", y = " ") +
      theme_classic(base_size = size) +
      theme(axis.text.x=element_blank(),
            axis.ticks.x = element_blank(),
            legend.title = element_blank(),
            legend.position = "none") +
      scale_color_wsu() +
      scale_fill_wsu()
    
    p_octmar_exceedance <- df %>% 
      filter(month >= 10 | month <= 3) %>%
      group_by(group, water_year, month) %>%
      summarise(combined = sum(combined)) %>% 
      group_by(group) %>% 
      mutate(prob = rank(combined)/(n()+1)) %>%
      summarise(combined_80 = nth(combined, which.min(abs(prob-0.8))),
                combined_90 = nth(combined, which.min(abs(prob-0.9))),
                combined_95 = nth(combined, which.min(abs(prob-0.95)))) %>%
      ggplot() +
      geom_col(aes(x = group, y = combined_80, color = group, fill = group), alpha = 0.33) +
      geom_col(aes(x = group, y = combined_90, color = group, fill = group), alpha = 0.33) +
      geom_col(aes(x = group, y = combined_95, color = group, fill = group), alpha = 0.33) +
      labs(x = " ", y = " ") +
      theme_classic(base_size = size) +
      theme(axis.text.x=element_blank(),
            axis.ticks.x = element_blank(),
            legend.title = element_blank(),
            legend.position = "none") +
      scale_color_wsu() +
      scale_fill_wsu()
    
    p_exceedance_m <- plot_grid(p_monthly_exceedance, p_yearly_exceedance, p_octmar_exceedance, rel_widths = c(6, 1, 1), align = 'h', nrow = 1, axis = "b")
    
  
    
    #### combined PLOT ####
    p_monthly_combined <- df %>% group_by(month, year, group) %>%
      summarise(total_combined = sum(combined)) %>%
      ggplot() +
      geom_boxplot(aes(x = group, y = total_combined, color = group, fill = group), alpha = 0.75) +
      # geom_jitter(aes(x = group, y = total_combined, color = group), alpha = 0.5, width = 0.05) +
      theme_classic(base_size = size) +
      facet_grid(.~month, switch = "x") +
      theme(axis.text.x = element_blank(),
            axis.ticks.x = element_blank(),
            legend.title = element_blank(),
            legend.position = "bottom",
            strip.background = element_blank()) +
      scale_color_wsu() +
      scale_fill_wsu() +
      labs(y = "combined", x = " ")
    
    p_yearly_combined <- df %>% group_by(year, group) %>%
      summarise(total_combined = sum(combined)) %>%
      ggplot() +
      geom_boxplot(aes(x = group, y = total_combined, color = group, fill = group), alpha = 0.75) +
      # geom_jitter(aes(x = group, y = total_combined, color = group), alpha = 0.5, width = 0.05) +
      theme_classic(base_size = size) +
      theme(axis.text.x = element_blank(),
            axis.ticks.x = element_blank(),
            legend.title = element_blank(),
            legend.position = "none") +
      scale_color_wsu() +
      scale_fill_wsu() +
      labs(y = " ", x = "Annual")
    
    p_octmar_combined <- df %>% mutate(water_year = year(time_stamp + month(3))) %>%
      group_by(group, water_year) %>%
      filter(month >= 10 | month <= 3) %>%
      summarise(total_combined = sum(combined)) %>%
      ggplot() +
      geom_boxplot(aes(x = group, y = total_combined, color = group, fill = group), alpha = 0.75) +
      # geom_jitter(aes(x = group, y = total_combined, color = group), alpha = 0.5, width = 0.05) +
      theme_classic(base_size = size) +
      theme(axis.text.x = element_blank(),
            axis.ticks.x = element_blank(),
            legend.title = element_blank(),
            legend.position = "none") +
      scale_color_wsu() +
      scale_fill_wsu() +
      labs(y = " ", x = "Oct - Mar")
    
    #### FINAL PLOT COMBINE ####
    
    legend <- get_legend(p_monthly_combined)
    
    p_combined <- plot_grid(p_monthly_combined + theme(legend.position = "none"), p_yearly_combined, p_octmar_combined, rel_widths = c(6, 1, 1), align = 'h', nrow = 1, axis = "b")
    
    p_grid <- plot_grid(p_prob_m, p_exceedance_m, p_combined, ncol = 1, align = "v", axis = 'l')
    
    plot_grid(p_grid, legend, ncol = 1, rel_heights = c(1,.05))
  

}
#############
############# storm_plots
#############
require(dplyr)
require(ggplot2)

source('functions/multiplot.R') # plots gridded figures
source("functions/storm_calc.R") ## calculates return intensity
storm_plots <- function(d){
  
  # p1 <- d %>% filter(!is.na(group)) %>% 
  #   storm_calc(1) %>%
  #   ggplot() +
  #   geom_line(aes(x = return_period, y = XT, color = group)) +
  #   labs(x = "Return Period (years)", y = "Storm Intensity (mm/hr)", title = "Max Precip") +
  #   theme_bw()
  
  # p2 <- d %>% filter(!is.na(group)) %>%
  #   storm_calc(.999) %>%
  #   ggplot() +
  #   geom_line(aes(x = return_period, y = XT, color = group)) +
  #   labs(x = "Return Period (years)", y = "Storm Intensity (mm/hr)", title = "99.9 Percentile Precip") +
  #   theme_bw()
  
  p3 <- d %>% filter(!is.na(group)) %>%
    storm_calc(.99) %>%
    ggplot() +
    geom_line(aes(x = return_period, y = XT, color = group)) +
    labs(x = "Return Period (years)", y = "Storm Intensity (mm/hr)", title = "A) 99 Percentile Precip") +
    theme_bw()
  
  p4 <- d %>%
    filter(precip >= quantile(precip, .999), !is.na(group)) %>%
    group_by(group) %>% 
    ggplot() +
    stat_ecdf(aes(x = precip, color = group), geom = "line") +
    labs(title = "B) Precip > 99.9 percentile") +
    theme_bw()
  
  p5 <- d %>%
    filter(precip >= quantile(precip, .95), !is.na(group)) %>%
    ggplot() +
    stat_ecdf(aes(x = precip, color = group), geom = "line") +
    labs(title = "C) Precip > 95th percentile") +
    theme_bw()
  
  p6 <- d %>%
    ggplot() +
    geom_line(aes(y = precip, x = time_stamp)) +
    geom_point(aes(y = precip, x = time_stamp)) +
    labs(title = "D)") +
    theme_bw()
  
  # p7 <- d %>% filter(month <= 4 | month >= 9) %>%
  #   group_by(water_year) %>%
  #   filter(precip > 0) %>%
  #   distinct(time_stamp, .keep_all = TRUE) %>%
  #   summarise(rain_days = n()) %>%
  #   ggplot() +
  #   geom_point(aes(x = water_year, y = rain_days)) +
  #   labs(x = "Water Year", y = "No. of days with Precip") +
  #   theme_bw()
  # 
  # p8 <- d %>% distinct(time_stamp, .keep_all = TRUE) %>%
  #   filter(!is.na(group)) %>% 
  #   group_by(month, group) %>%
  #   summarise(month_precip = mean(precip)) %>%
  #   ggplot() +
  #   geom_line(aes(x = month, y = month_precip, color = group)) +
  #   labs(x = "Month", y = "Mean Monthly Precip (mm)") +
  #   theme_bw()
  # 
  # p9 <- d %>% filter(!is.na(group)) %>%
  #   ggplot() +
  #   geom_boxplot(aes(y = precip, x = group)) +
  #   scale_y_log10() +
  #   theme_bw()
  # 
  p10 <- d %>% filter(precip >= quantile(precip, .995), !is.na(group)) %>% 
    group_by(precip, group) %>% 
    summarise(days = n()) %>% 
    arrange(desc(precip)) %>% 
    group_by(group) %>% 
    mutate(days_above = cumsum(days)) %>% 
    ggplot() +
    geom_line(aes(x = precip, y = days_above, color = group)) +
    labs(title = "E) Precip > 99.5%") +
    theme_bw()
  
  p11 <- d %>% filter(precip >= quantile(precip, .95), !is.na(group)) %>% 
    group_by(precip, group) %>% 
    summarise(days = n()) %>% 
    arrange(desc(precip)) %>% 
    group_by(group) %>% 
    mutate(days_above = cumsum(days)) %>% 
    ggplot() +
    geom_line(aes(x = precip, y = days_above, color = group)) +
    labs(title = "F) Precip > 95%") +
    theme_bw()
  
  p12 <- d %>% mutate(precip_sum = stats::filter(precip,rep(1, 5), sides=2)) %>% 
    ggplot() +
    geom_line(aes(x = time_stamp, y = precip_sum)) +
    labs(title = "G) 5 Day window sum precip") +
    theme_bw()
  
  p13 <- d %>% mutate(precip = stats::filter(precip,rep(1, 5), sides=2)) %>% 
    filter(!is.na(group), !is.na(precip)) %>%
    storm_calc(.99) %>%
    ggplot() +
    geom_line(aes(x = return_period, y = XT, color = group)) +
    labs(x = "Return Period (years)", y = "Storm Intensity (mm/hr)", title = "h) 99 Percentile Precip on 5 day window") +
    theme_bw()
  
  p14 <- d %>% group_by(month, year) %>% 
    mutate(cum_precip = cumsum(precip)) %>% 
    ggplot() +
    geom_line(aes(x = time_stamp, y = cum_precip)) +
    theme_minimal()
  
  multiplot(p3, p4, p5, p6, p10, p11, p12, p13, p14, cols = 3)
}
#############


#############
############# plot_dry_days
#############
library(dplyr)
library(forcats)
library(ggplot2)
library(cowplot)

plot_drydays_boxplot <- function(df, sel_climate_proj){
  
  df %>% 
    mutate(group = fct_relevel(as_factor(group), c("hist", "2040s", "2060s", "2080s")),
           season = case_when(month <= 3 ~ "Jan-Mar",
                              between(month, 4, 6) ~ "Apr-Jun",
                              between(month, 7, 9) ~ "Jul-Sep",
                              month >= 10 ~ "Oct-Dec")) %>% 
    filter(precip <= quantile(precip, 0.05, na.rm = TRUE), !is.na(group)) %>% 
    group_by(climate_proj, model, group, season, year) %>% 
    summarise(dry_days = n()) %>%
    # summarise(dry_days = median(dry_days)) %>%
    filter(climate_proj %in% c(NA, sel_climate_proj)) %>% 
    ggplot() +
    geom_boxplot(aes(x = group, y = dry_days)) +
    geom_jitter(aes(x = group, y = dry_days), width = 0.2, alpha = 0.2) +
    facet_grid(season~., scales = "free_y") +
    labs(y = "Number of Dry Days", title  = "Days under 5 Percentile of Precipitation") +
    theme(axis.title.x = element_blank())  
}
#******************************************************

#############
############# print_prob_plots
#############
#### Print Probability Plots ####

print_prob_plots <- function(df){
  # Plot Size
  size <- 16
  
  #### DAILY PROB ####
  df_monthly_exceedance <- df %>% 
                           filter(group == 'hist') %>% 
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
  
  df_octmar_exceedance <- df %>% 
                          filter(group == 'hist', month >= 10 | month <= 3) %>%
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
  
  p_prob_daily <- plot_grid(p_monthly_exceedance_val,p_octmar_exceedance_val, 
                            nrow = 1, align = "vh", 
                            rel_widths = c(3.25, 1), 
                            axis = 'b')
  
  # save_plot("figures/daily_exceedance_probability_dry.pdf", p_prob, base_aspect_ratio = 2, scale = 1, limitsize = FALSE)
  
  #### 7 DAY PROB ####
  
  df_7 <- df %>% 
          group_by(group, model) 
          %>% mutate(precip = rollsum(precip, 7, align = "right", fill = NA))
  
  df_monthly_exceedance <- df_7 %>% 
                           filter(group == 'hist') %>% 
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
  
  df_octmar_exceedance <- df_7 %>% 
                          filter(group == 'hist', month >= 10 | month <= 3) %>%
                          mutate(prob = rank(precip)/(n()+1)) %>%
                          summarise(precip_80 = nth(precip, which.min(abs(prob-0.8))),
                                    precip_90 = nth(precip, which.min(abs(prob-0.9))),
                                    precip_95 = nth(precip, which.min(abs(prob-0.95)))
                                    )
  
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
  
  p_prob_7day <- plot_grid(p_monthly_exceedance_val,
                           p_octmar_exceedance_val, 
                           nrow = 1, align = "vh", 
                           rel_widths = c(3.25, 1), axis = 'b')
  
  # save_plot("figures/7_day_exceedance_probability_dry.pdf", 
  #           p_prob, base_aspect_ratio = 2, scale = 1, 
  #           limitsize = FALSE)
  #### MONTH PROB ####
  
  df_monthly_exceedance <- df %>% 
                           filter(group == 'hist') %>% 
                           group_by(month, year) %>%
                           summarise(precip = sum(precip)) %>% 
                           group_by(month) %>%
                           mutate(prob = rank(precip)/(n()+1)) %>%
                           summarise(precip_80 = nth(precip, which.min(abs(prob-0.8))),
                                     precip_90 = nth(precip, which.min(abs(prob-0.9))),
                                     precip_95 = nth(precip, which.min(abs(prob-0.95))))
  
  df_monthly_prob <- full_join(df, df_monthly_exceedance, by = c("month")) %>% 
                     filter(!is.na(group))
  
  p_monthly_exceedance_val <- df %>% 
                              filter(group != "hist", month >= 10 | month <= 3) %>% 
                              group_by(group, month, climate_proj, model, year) %>%
                              summarise(precip = sum(precip)) %>% 
                              full_join(df_monthly_exceedance, by = "month") %>% 
                              group_by(group, month, model, climate_proj) %>% 
                              mutate(prob = rank(-precip)/(n()+1)) %>%
                              summarise(precip_80 = first(precip_80),
                                        precip_90 = first(precip_90),
                                        precip_95 = first(precip_95),
                                        prob_80 = nth(prob, which.min(abs(precip-precip_80))),
                                        prob_90 = nth(prob, which.min(abs(precip-precip_90))),
                                        prob_95 = nth(prob, which.min(abs(precip-precip_95)))) %>%
                              gather(exceedance, prob, -model, -month, -group) %>% 
                              group_by(group, month, exceedance) %>% 
                              mutate(prob_median = median(prob),
                                     hist_prob = case_when(exceedance == "prob_80" ~ 0.20,
                                                           exceedance == "prob_90" ~ 0.10,
                                                           exceedance == "prob_95" ~ 0.05)) %>% 
                              plot_monthly_prob("Monthly Prob")
  
  df_octmar_exceedance <- df %>% 
                          filter(group == 'hist', month >= 10 | month <= 3) %>%
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
  
  p_prob_monthly <- plot_grid(p_monthly_exceedance_val, 
                              p_octmar_exceedance_val, 
                              nrow = 1, align = "vh", 
                              rel_widths = c(3.25, 1), axis = 'b')
  
  p_all <- plot_grid(p_prob_daily, p_prob_7day, p_prob_monthly, ncol = 1, align = "vh")
  
  return(p_all) 
}
#******************************************************

#############
############# probability_plots
#############
# size <- 20
# scaleFUN <- function(x) sprintf("%.2f", x)
plot_monthly_prob <- function(df, label){
  
  ggplot(df) +
    geom_jitter(aes(x = group, y = prob, color = group, fill = group), width = .2) +
    geom_line(aes(x = group, y = prob_median, group = model, color = "hist")) +
    geom_hline(aes(yintercept = hist_prob), color = "grey40", linetype = "longdash") +
    facet_grid(exceedance~month, switch = "x", scales = "free_y") +
    labs(y = label) +
    theme_linedraw() +
    theme(axis.text.x = element_blank(), 
          axis.ticks.x = element_blank(),
          axis.title.x = element_blank(),
          legend.title = element_blank(), 
          legend.position = "none",
          strip.background = element_blank(), 
          strip.text.y = element_blank(),
          strip.placement = "outside",
          panel.spacing.y = unit(.75, "lines"),
          strip.text = element_text(colour = 'black')) +
    scale_color_wsu(palette = 'rev') +
    scale_fill_wsu(palette = 'rev') +
    # scale_color_viridis_d(option = "plasma") +
    # scale_fill_viridis_d(option = "plasma") +
    scale_y_continuous(labels=scales::percent)
}

plot_octmar_prob <- function(df){
  
  ggplot(df) +
    geom_jitter(aes(x = group, y = prob, color = group), width = .2) +
    geom_line(aes(x = group, y = prob_median, group = model, color = "hist")) +
    geom_hline(aes(yintercept = hist_prob), color = "grey40", linetype = "longdash") +
    facet_grid(exceedance~., switch = "x", scales = "free_y") +
    labs(x = "Oct - Mar") +
    theme_linedraw() +
    theme(axis.text.x = element_blank(),
          axis.ticks.x = element_blank(),
          axis.title.y = element_blank(),
          legend.title = element_blank(),
          legend.position = "none",
          strip.background = element_blank(),
          axis.title.x = element_text(size = 8, face = "plain"),
          panel.spacing.y = unit(.75, "lines")) +
    scale_color_wsu(palette = 'rev') +
    scale_fill_wsu(palette = 'rev') +
    # scale_color_viridis_d(option = "plasma") +
    # scale_fill_viridis_d(option = "plasma") +
    scale_y_continuous(labels=scales::percent)
}

#******************************************************

#############
############# multiplot
#############

# Multiple plot function
#
# ggplot objects can be passed in ..., or to plotlist (as a list of ggplot objects)
# - cols:   Number of columns in layout
# - layout: A matrix specifying the layout. If present, 'cols' is ignored.
#
# If the layout is something like matrix(c(1,2,3,3), nrow=2, byrow=TRUE),
# then plot 1 will go in the upper left, 2 will go in the upper right, and
# 3 will go all the way across the bottom.
#
multiplot <- function(..., plotlist=NULL, file, cols=1, layout=NULL) {
  library(grid)
  
  # Make a list from the ... arguments and plotlist
  plots <- c(list(...), plotlist)
  
  numPlots = length(plots)
  
  # If layout is NULL, then use 'cols' to determine layout
  if (is.null(layout)) {
    # Make the panel
    # ncol: Number of columns of plots
    # nrow: Number of rows needed, calculated from # of cols
    layout <- matrix(seq(1, cols * ceiling(numPlots/cols)),
                     ncol = cols, nrow = ceiling(numPlots/cols))
  }
  
  if (numPlots==1) {
    print(plots[[1]])
    
  } else {
    # Set up the page
    grid.newpage()
    pushViewport(viewport(layout = grid.layout(nrow(layout), ncol(layout))))
    
    # Make each plot, in the correct location
    for (i in 1:numPlots) {
      # Get the i,j matrix positions of the regions that contain this subplot
      matchidx <- as.data.frame(which(layout == i, arr.ind = TRUE))
      
      print(plots[[i]], vp = viewport(layout.pos.row = matchidx$row,
                                      layout.pos.col = matchidx$col))
    }
  }
}

#******************************************************
#############
############# precip_plots
#############

precip_plots <- function(df){
  
  #### INIT CALCS ####
  
  df <- filter(df, !is.na(group))
  df <- mutate(df, water_year = year(time_stamp + month(3)),
               month_name = month(time_stamp, label = TRUE))

  # set plot scale
  size <- 14
  
  #### PROBABILITY PLOT DAILY ####
  
  # Calculate monthly exceedance
  df_monthly_exceedance <- df %>% filter(group == 'hist') %>% 
    group_by(month) %>%
    mutate(prob = rank(precip)/(n()+1)) %>%
    summarise(precip_80 = nth(precip, which.min(abs(prob-0.8))),
              precip_90 = nth(precip, which.min(abs(prob-0.9))),
              precip_95 = nth(precip, which.min(abs(prob-0.95))))
  
  df_monthly_prob <- full_join(df, df_monthly_exceedance, by = "month")
  
  p_monthly_exceedance_val <- df_monthly_prob %>% 
    group_by(group, month) %>%
    mutate(prob = rank(-precip)/(n()+1)) %>%
    summarise(prob_80 = nth(prob, which.min(abs(precip-precip_80))),
              prob_90 = nth(prob, which.min(abs(precip-precip_90))),
              prob_95 = nth(prob, which.min(abs(precip-precip_95))),
              month_name = first(month_name)) %>%
    ggplot() +
    geom_col(aes(x = group, y = prob_80, color = group, fill = group), alpha = 0.33) +
    geom_col(aes(x = group, y = prob_90, color = group, fill = group), alpha = 0.33) +
    geom_col(aes(x = group, y = prob_95, color = group, fill = group), alpha = 0.33) +
    facet_grid(.~month_name, switch = "x") +
    labs(x = " ", y = "Daily Prob") +
    theme_classic(base_size = size) +
    theme(axis.text.x=element_blank(), 
          axis.ticks.x = element_blank(),
          legend.title = element_blank(), 
          legend.position = "none",
          strip.background = element_blank(),
          strip.text = element_blank()) +
    scale_color_wsu() +
    scale_fill_wsu()
  
  df_yearly_exceedance <- df %>% filter(group == 'hist') %>% 
    mutate(prob = rank(precip)/(n()+1)) %>%
    summarise(precip_80 = nth(precip, which.min(abs(prob-0.8))),
              precip_90 = nth(precip, which.min(abs(prob-0.9))),
              precip_95 = nth(precip, which.min(abs(prob-0.95))))
  
  
  p_yearly_exceedance_val <- df %>% 
    group_by(group) %>%
    mutate(prob = rank(-precip)/(n()+1)) %>%
    summarise(prob_80 = nth(prob, which.min(abs(precip-df_yearly_exceedance$precip_80))),
              prob_90 = nth(prob, which.min(abs(precip-df_yearly_exceedance$precip_90))),
              prob_95 = nth(prob, which.min(abs(precip-df_yearly_exceedance$precip_95)))) %>%
    ggplot() +
    geom_col(aes(x = group, y = prob_80, color = group, fill = group), alpha = 0.33) +
    geom_col(aes(x = group, y = prob_90, color = group, fill = group), alpha = 0.33) +
    geom_col(aes(x = group, y = prob_95, color = group, fill = group), alpha = 0.33) +
    labs(x = " ", y = " ") +
    theme_classic(base_size = size) +
    theme(axis.text.x=element_blank(), 
          axis.ticks.x = element_blank(),
          legend.title = element_blank(), 
          legend.position = "none",
          strip.background = element_blank(),
          strip.text = element_blank()) +
    scale_color_wsu() +
    scale_fill_wsu()
  
  df_octmar_exceedance <- df %>% filter(group == 'hist', month >= 10 | month <= 3) %>%
    mutate(prob = rank(precip)/(n()+1)) %>%
    summarise(precip_80 = nth(precip, which.min(abs(prob-0.8))),
              precip_90 = nth(precip, which.min(abs(prob-0.9))),
              precip_95 = nth(precip, which.min(abs(prob-0.95))))

  p_octmar_exceedance_val <- df %>%
    filter(month >= 10 | month <= 3) %>% 
    group_by(group) %>%
    mutate(prob = rank(-precip)/(n()+1)) %>%
    summarise(prob_80 = nth(prob, which.min(abs(precip-df_octmar_exceedance$precip_80))),
              prob_90 = nth(prob, which.min(abs(precip-df_octmar_exceedance$precip_90))),
              prob_95 = nth(prob, which.min(abs(precip-df_octmar_exceedance$precip_95)))) %>%
    ggplot() +
    geom_col(aes(x = group, y = prob_80, color = group, fill = group), alpha = 0.33) +
    geom_col(aes(x = group, y = prob_90, color = group, fill = group), alpha = 0.33) +
    geom_col(aes(x = group, y = prob_95, color = group, fill = group), alpha = 0.33) +
    labs(x = " ", y = " ") +
    theme_classic(base_size = size) +
    theme(axis.text.x=element_blank(),
          axis.ticks.x = element_blank(),
          legend.title = element_blank(),
          legend.position = "none",
          strip.background = element_blank(),
          strip.text = element_blank()) +
    scale_color_wsu() +
    scale_fill_wsu()
  
  p_prob <- plot_grid(p_monthly_exceedance_val, p_yearly_exceedance_val,p_octmar_exceedance_val, nrow = 1, align = "h", rel_widths = c(6, 1, 1), axis = 'b')
  
  #### PROBABILITY PLOT MONTHLY ####
  
  # Calculate monthly exceedance
  df_monthly_exceedance <- df %>% filter(group == 'hist') %>% 
    group_by(month, year) %>%
    summarise(precip = sum(precip)) %>% 
    group_by(month) %>% 
    mutate(prob = rank(precip)/(n()+1)) %>%
    summarise(precip_80 = nth(precip, which.min(abs(prob-0.8))),
              precip_90 = nth(precip, which.min(abs(prob-0.9))),
              precip_95 = nth(precip, which.min(abs(prob-0.95))))
  
  df_monthly_prob <- full_join(df, df_monthly_exceedance, by = "month")
  
  p_monthly_exceedance_val <- df_monthly_prob %>% 
    group_by(group, month, year) %>% 
    summarise(precip = sum(precip),
              precip_80 = first(precip_80),
              precip_90 = first(precip_90),
              precip_95 = first(precip_95)) %>% 
    group_by(group, month) %>% 
    mutate(prob = rank(-precip)/(n()+1)) %>%
    summarise(prob_80 = nth(prob, which.min(abs(precip-precip_80))),
              prob_90 = nth(prob, which.min(abs(precip-precip_90))),
              prob_95 = nth(prob, which.min(abs(precip-precip_95)))) %>%
    ggplot() +
    geom_col(aes(x = group, y = prob_80, color = group, fill = group), alpha = 0.33) +
    geom_col(aes(x = group, y = prob_90, color = group, fill = group), alpha = 0.33) +
    geom_col(aes(x = group, y = prob_95, color = group, fill = group), alpha = 0.33) +
    facet_grid(.~month, switch = "x") +
    labs(x = " ", y = "Monthly Prob") +
    theme_classic(base_size = size) +
    theme(axis.text.x=element_blank(), 
          axis.ticks.x = element_blank(),
          legend.title = element_blank(), 
          legend.position = "none",
          strip.background = element_blank(),
          strip.text = element_blank()) +
    scale_color_wsu() +
    scale_fill_wsu()
  
  df_yearly_exceedance <- df %>% filter(group == 'hist') %>% 
    group_by(month, year) %>% 
    summarise(precip = sum(precip)) %>% 
    ungroup() %>% 
    mutate(prob = rank(precip)/(n()+1)) %>%
    summarise(precip_80 = nth(precip, which.min(abs(prob-0.8))),
              precip_90 = nth(precip, which.min(abs(prob-0.9))),
              precip_95 = nth(precip, which.min(abs(prob-0.95))))
  
  
  p_yearly_exceedance_val <- df %>% 
    group_by(month, group, year) %>% 
    summarise(precip = sum(precip)) %>% 
    group_by(group) %>% 
    mutate(prob = rank(-precip)/(n()+1)) %>%
    summarise(prob_80 = nth(prob, which.min(abs(precip-df_yearly_exceedance$precip_80))),
              prob_90 = nth(prob, which.min(abs(precip-df_yearly_exceedance$precip_90))),
              prob_95 = nth(prob, which.min(abs(precip-df_yearly_exceedance$precip_95)))) %>%
    ggplot() +
    # geom_col(aes(x = group, y = prob_80, color = group, fill = group), alpha = 0.33) +
    # geom_col(aes(x = group, y = prob_90, color = group, fill = group), alpha = 0.33) +
    # geom_col(aes(x = group, y = prob_95, color = group, fill = group), alpha = 0.33) +
    geom_col(aes(x = group, y = prob_80, color = group, fill = group), alpha = 0.33) +
    geom_col(aes(x = group, y = prob_90, color = group, fill = group), alpha = 0.33) +
    geom_col(aes(x = group, y = prob_95, color = group, fill = group), alpha = 0.33) +
    labs(x = " ", y = " ") +
    theme_classic(base_size = size) +
    theme(axis.text.x=element_blank(), 
          axis.ticks.x = element_blank(),
          legend.title = element_blank(), 
          legend.position = "none",
          strip.background = element_blank(),
          strip.text = element_blank()) +
    scale_color_wsu() +
    scale_fill_wsu()
  
  df_octmar_exceedance <- df %>% filter(group == 'hist', month >= 10 | month <= 3) %>%
    group_by(month, group, water_year) %>% 
    summarise(precip = sum(precip)) %>% 
    ungroup() %>% 
    mutate(prob = rank(precip)/(n()+1)) %>%
    summarise(precip_80 = nth(precip, which.min(abs(prob-0.8))),
              precip_90 = nth(precip, which.min(abs(prob-0.9))),
              precip_95 = nth(precip, which.min(abs(prob-0.95))))
  
  p_octmar_exceedance_val <- df %>% filter(month >= 10 | month <= 3) %>%
    group_by(month, group, year) %>% 
    summarise(precip = sum(precip)) %>% 
    group_by(group) %>% 
    mutate(prob = rank(-precip)/(n()+1)) %>%
    summarise(prob_80 = nth(prob, which.min(abs(precip-df_octmar_exceedance$precip_80))),
              prob_90 = nth(prob, which.min(abs(precip-df_octmar_exceedance$precip_90))),
              prob_95 = nth(prob, which.min(abs(precip-df_octmar_exceedance$precip_95)))) %>%
    ggplot() +
    geom_col(aes(x = group, y = prob_80, color = group, fill = group), alpha = 0.33) +
    geom_col(aes(x = group, y = prob_90, color = group, fill = group), alpha = 0.33) +
    geom_col(aes(x = group, y = prob_95, color = group, fill = group), alpha = 0.33) +
    labs(x = " ", y = " ") +
    theme_classic(base_size = size) +
    theme(axis.text.x=element_blank(),
          axis.ticks.x = element_blank(),
          legend.title = element_blank(),
          legend.position = "none",
          strip.background = element_blank(),
          strip.text = element_blank()) +
    scale_color_wsu() +
    scale_fill_wsu()
  
  p_prob_m <- plot_grid(p_monthly_exceedance_val, p_yearly_exceedance_val,p_octmar_exceedance_val, nrow = 1, align = "h", rel_widths = c(6, 1, 1), axis = 'b')
  
  #### PROBABILITY PLOT 7 DAY WINDOW ####
  
  # Calculate monthly exceedance
  
  df_7 <- df %>% group_by(group) %>% mutate(precip = rollsum(precip, 7, align = "right", fill = NA))
  
  df_monthly_exceedance <- df_7 %>% filter(group == 'hist') %>% 
    group_by(month) %>%
    mutate(prob = rank(precip)/(n()+1)) %>%
    summarise(precip_80 = nth(precip, which.min(abs(prob-0.8))),
              precip_90 = nth(precip, which.min(abs(prob-0.9))),
              precip_95 = nth(precip, which.min(abs(prob-0.95))))
  
  df_monthly_prob <- full_join(df_7, df_monthly_exceedance, by = "month")
  
  
  p_monthly_exceedance_val <- df_monthly_prob %>% 
    group_by(group, month) %>%
    mutate(prob = rank(-precip)/(n()+1)) %>%
    summarise(prob_80 = nth(prob, which.min(abs(precip-precip_80))),
              prob_90 = nth(prob, which.min(abs(precip-precip_90))),
              prob_95 = nth(prob, which.min(abs(precip-precip_95))),
              month_name = first(month_name)) %>%
    ggplot() +
    geom_col(aes(x = group, y = prob_80, color = group, fill = group), alpha = 0.33) +
    geom_col(aes(x = group, y = prob_90, color = group, fill = group), alpha = 0.33) +
    geom_col(aes(x = group, y = prob_95, color = group, fill = group), alpha = 0.33) +
    facet_grid(.~month_name, switch = "x") +
    labs(x = " ", y = "7 day Prob") +
    theme_classic(base_size = size) +
    theme(axis.text.x=element_blank(), 
          axis.ticks.x = element_blank(),
          legend.title = element_blank(), 
          legend.position = "none",
          strip.background = element_blank(),
          strip.text = element_blank()) +
    scale_color_wsu() +
    scale_fill_wsu()
  
  df_yearly_exceedance <- df_7 %>% filter(group == 'hist') %>% 
    mutate(prob = rank(precip)/(n()+1)) %>%
    summarise(precip_80 = nth(precip, which.min(abs(prob-0.8))),
              precip_90 = nth(precip, which.min(abs(prob-0.9))),
              precip_95 = nth(precip, which.min(abs(prob-0.95))))
  
  
  p_yearly_exceedance_val <- df_7 %>% 
    group_by(group) %>%
    mutate(prob = rank(-precip)/(n()+1)) %>%
    summarise(prob_80 = nth(prob, which.min(abs(precip-df_yearly_exceedance$precip_80))),
              prob_90 = nth(prob, which.min(abs(precip-df_yearly_exceedance$precip_90))),
              prob_95 = nth(prob, which.min(abs(precip-df_yearly_exceedance$precip_95)))) %>%
    ggplot() +
    geom_col(aes(x = group, y = prob_80, color = group, fill = group), alpha = 0.33) +
    geom_col(aes(x = group, y = prob_90, color = group, fill = group), alpha = 0.33) +
    geom_col(aes(x = group, y = prob_95, color = group, fill = group), alpha = 0.33) +
    labs(x = " ", y = " ") +
    theme_classic(base_size = size) +
    theme(axis.text.x=element_blank(), 
          axis.ticks.x = element_blank(),
          legend.title = element_blank(), 
          legend.position = "none",
          strip.background = element_blank(),
          strip.text = element_blank()) +
    scale_color_wsu() +
    scale_fill_wsu()
  
  df_octmar_exceedance <- df_7 %>% filter(group == 'hist', month >= 10 | month <= 3) %>%
    mutate(prob = rank(precip)/(n()+1)) %>%
    summarise(precip_80 = nth(precip, which.min(abs(prob-0.8))),
              precip_90 = nth(precip, which.min(abs(prob-0.9))),
              precip_95 = nth(precip, which.min(abs(prob-0.95))))
  
  p_octmar_exceedance_val <- df_7 %>%
    filter(month >= 10 | month <= 3) %>% 
    group_by(group) %>%
    mutate(prob = rank(-precip)/(n()+1)) %>%
    summarise(prob_80 = nth(prob, which.min(abs(precip-df_octmar_exceedance$precip_80))),
              prob_90 = nth(prob, which.min(abs(precip-df_octmar_exceedance$precip_90))),
              prob_95 = nth(prob, which.min(abs(precip-df_octmar_exceedance$precip_95)))) %>%
    ggplot() +
    geom_col(aes(x = group, y = prob_80, color = group, fill = group), alpha = 0.33) +
    geom_col(aes(x = group, y = prob_90, color = group, fill = group), alpha = 0.33) +
    geom_col(aes(x = group, y = prob_95, color = group, fill = group), alpha = 0.33) +
    labs(x = " ", y = " ") +
    theme_classic(base_size = size) +
    theme(axis.text.x=element_blank(),
          axis.ticks.x = element_blank(),
          legend.title = element_blank(),
          legend.position = "none",
          strip.background = element_blank(),
          strip.text = element_blank()) +
    scale_color_wsu() +
    scale_fill_wsu()
  
  p_prob_7 <- plot_grid(p_monthly_exceedance_val, p_yearly_exceedance_val,p_octmar_exceedance_val, nrow = 1, align = "h", rel_widths = c(6, 1, 1), axis = 'b')
  
  #### EXCEEDANCE PLOT DAILY ####
  
  p_monthly_exceedance <- df %>% group_by(group, month, year) %>%
    mutate(prob = rank(precip)/(n()+1)) %>%
    group_by(group, month) %>%
    summarise(precip_80 = nth(precip, which.min(abs(prob-0.8))),
              precip_90 = nth(precip, which.min(abs(prob-0.9))),
              precip_95 = nth(precip, which.min(abs(prob-0.95))),
              month_name = month(first(time_stamp), label = TRUE)) %>%
    ggplot() +
    geom_col(aes(x = group, y = precip_80, color = group, fill = group), alpha = 0.33) +
    geom_col(aes(x = group, y = precip_90, color = group, fill = group), alpha = 0.33) +
    geom_col(aes(x = group, y = precip_95, color = group, fill = group), alpha = 0.33) +
    facet_grid(.~month_name, switch = "x") +
    labs(x = " ", y = "Daily Exceed") +
    theme_classic(base_size = size) +
    theme(axis.text.x=element_blank(),
          axis.ticks.x = element_blank(),
          legend.title = element_blank(),
          legend.position = "none",
          strip.background = element_blank(),
          strip.text = element_blank()) +
    scale_color_wsu() +
    scale_fill_wsu()

  p_yearly_exceedance <- df %>% group_by(group) %>%
    mutate(prob = rank(precip)/(n()+1)) %>%
    summarise(precip_80 = nth(precip, which.min(abs(prob-0.8))),
              precip_90 = nth(precip, which.min(abs(prob-0.9))),
              precip_95 = nth(precip, which.min(abs(prob-0.95)))) %>%
    ggplot() +
    geom_col(aes(x = group, y = precip_80, color = group, fill = group), alpha = 0.33) +
    geom_col(aes(x = group, y = precip_90, color = group, fill = group), alpha = 0.33) +
    geom_col(aes(x = group, y = precip_95, color = group, fill = group), alpha = 0.33) +
    # facet_grid(.~month_name, switch = "x") +
    labs(x = " ", y = " ") +
    theme_classic(base_size = size) +
    theme(axis.text.x=element_blank(),
          axis.ticks.x = element_blank(),
          legend.title = element_blank(),
          legend.position = "none") +
    scale_color_wsu() +
    scale_fill_wsu()

  p_octmar_exceedance <- df %>%
    group_by(group, water_year) %>%
    filter(month >= 10 | month <= 3) %>%
    mutate(prob = rank(precip)/(n()+1)) %>%
    group_by(group) %>%
    summarise(precip_80 = nth(precip, which.min(abs(prob-0.8))),
              precip_90 = nth(precip, which.min(abs(prob-0.9))),
              precip_95 = nth(precip, which.min(abs(prob-0.95)))) %>%
    ggplot() +
    geom_col(aes(x = group, y = precip_80, color = group, fill = group), alpha = 0.33) +
    geom_col(aes(x = group, y = precip_90, color = group, fill = group), alpha = 0.33) +
    geom_col(aes(x = group, y = precip_95, color = group, fill = group), alpha = 0.33) +
    # facet_grid(.~month_name, switch = "x") +
    labs(x = " ", y = " ") +
    theme_classic(base_size = size) +
    theme(axis.text.x=element_blank(),
          axis.ticks.x = element_blank(),
          legend.title = element_blank(),
          legend.position = "none") +
    scale_color_wsu() +
    scale_fill_wsu()

  p_exceedance <- plot_grid(p_monthly_exceedance, p_yearly_exceedance, p_octmar_exceedance, rel_widths = c(6, 1, 1), align = 'h', nrow = 1, axis = "b")

  
  #### EXCEEDANCE PLOT MONTHLY ####
  
  p_monthly_exceedance <- df %>% group_by(group, month, year) %>%
    summarise(precip = sum(precip)) %>% 
    group_by(group, month) %>% 
    mutate(prob = rank(precip)/(n()+1)) %>%
    summarise(precip_80 = nth(precip, which.min(abs(prob-0.8))),
              precip_90 = nth(precip, which.min(abs(prob-0.9))),
              precip_95 = nth(precip, which.min(abs(prob-0.95)))) %>%
    ggplot() +
    geom_col(aes(x = group, y = precip_80, color = group, fill = group), alpha = 0.33) +
    geom_col(aes(x = group, y = precip_90, color = group, fill = group), alpha = 0.33) +
    geom_col(aes(x = group, y = precip_95, color = group, fill = group), alpha = 0.33) +
    facet_grid(.~month, switch = "x") +
    labs(x = " ", y = "Month Exceed") +
    theme_classic(base_size = size) +
    theme(axis.text.x=element_blank(),
          axis.ticks.x = element_blank(),
          legend.title = element_blank(),
          legend.position = "none",
          strip.background = element_blank(),
          strip.text = element_blank()) +
    scale_color_wsu() +
    scale_fill_wsu()

  p_yearly_exceedance <- df %>% group_by(group, month, water_year) %>%
    summarise(precip = sum(precip)) %>% 
    group_by(group) %>% 
    mutate(prob = rank(precip)/(n()+1)) %>%
    summarise(precip_80 = nth(precip, which.min(abs(prob-0.8))),
              precip_90 = nth(precip, which.min(abs(prob-0.9))),
              precip_95 = nth(precip, which.min(abs(prob-0.95)))) %>%
    ggplot() +
    geom_col(aes(x = group, y = precip_80, color = group, fill = group), alpha = 0.33) +
    geom_col(aes(x = group, y = precip_90, color = group, fill = group), alpha = 0.33) +
    geom_col(aes(x = group, y = precip_95, color = group, fill = group), alpha = 0.33) +
    # facet_grid(.~month_name, switch = "x") +
    labs(x = " ", y = " ") +
    theme_classic(base_size = size) +
    theme(axis.text.x=element_blank(),
          axis.ticks.x = element_blank(),
          legend.title = element_blank(),
          legend.position = "none") +
    scale_color_wsu() +
    scale_fill_wsu()

  p_octmar_exceedance <- df %>% 
    filter(month >= 10 | month <= 3) %>%
    group_by(group, water_year, month) %>%
    summarise(precip = sum(precip)) %>% 
    group_by(group) %>% 
    mutate(prob = rank(precip)/(n()+1)) %>%
    summarise(precip_80 = nth(precip, which.min(abs(prob-0.8))),
              precip_90 = nth(precip, which.min(abs(prob-0.9))),
              precip_95 = nth(precip, which.min(abs(prob-0.95)))) %>%
    ggplot() +
    geom_col(aes(x = group, y = precip_80, color = group, fill = group), alpha = 0.33) +
    geom_col(aes(x = group, y = precip_90, color = group, fill = group), alpha = 0.33) +
    geom_col(aes(x = group, y = precip_95, color = group, fill = group), alpha = 0.33) +
    labs(x = " ", y = " ") +
    theme_classic(base_size = size) +
    theme(axis.text.x=element_blank(),
          axis.ticks.x = element_blank(),
          legend.title = element_blank(),
          legend.position = "none") +
    scale_color_wsu() +
    scale_fill_wsu()

  p_exceedance_m <- plot_grid(p_monthly_exceedance, p_yearly_exceedance, p_octmar_exceedance, rel_widths = c(6, 1, 1), align = 'h', nrow = 1, axis = "b")

  
  #### EXCEEDANCE PLOT 7 DAY WINDOW ####
  
  p_monthly_exceedance <- df %>% group_by(group) %>% 
    mutate(precip = rollsum(precip, 7, align = "right", fill = NA)) %>% 
    group_by(group, month, year) %>%
    mutate(prob = rank(precip)/(n()+1)) %>%
    group_by(group, month) %>%
    summarise(precip_80 = nth(precip, which.min(abs(prob-0.8))),
              precip_90 = nth(precip, which.min(abs(prob-0.9))),
              precip_95 = nth(precip, which.min(abs(prob-0.95))),
              month_name = month(first(time_stamp), label = TRUE)) %>%
    ggplot() +
    geom_col(aes(x = group, y = precip_80, color = group, fill = group), alpha = 0.33) +
    geom_col(aes(x = group, y = precip_90, color = group, fill = group), alpha = 0.33) +
    geom_col(aes(x = group, y = precip_95, color = group, fill = group), alpha = 0.33) +
    facet_grid(.~month_name, switch = "x") +
    labs(x = " ", y = "7 Day Exceed") +
    theme_classic(base_size = size) +
    theme(axis.text.x=element_blank(),
          axis.ticks.x = element_blank(),
          legend.title = element_blank(),
          legend.position = "none",
          strip.background = element_blank(),
          strip.text = element_blank()) +
    scale_color_wsu() +
    scale_fill_wsu()
  
  p_yearly_exceedance <- df %>% group_by(group) %>% 
    mutate(precip = rollsum(precip, 7, align = "right", fill = NA),
           prob = rank(precip)/(n()+1)) %>%
    summarise(precip_80 = nth(precip, which.min(abs(prob-0.8))),
              precip_90 = nth(precip, which.min(abs(prob-0.9))),
              precip_95 = nth(precip, which.min(abs(prob-0.95)))) %>%
    ggplot() +
    geom_col(aes(x = group, y = precip_80, color = group, fill = group), alpha = 0.33) +
    geom_col(aes(x = group, y = precip_90, color = group, fill = group), alpha = 0.33) +
    geom_col(aes(x = group, y = precip_95, color = group, fill = group), alpha = 0.33) +
    # facet_grid(.~month_name, switch = "x") +
    labs(x = " ", y = " ") +
    theme_classic(base_size = size) +
    theme(axis.text.x=element_blank(),
          axis.ticks.x = element_blank(),
          legend.title = element_blank(),
          legend.position = "none") +
    scale_color_wsu() +
    scale_fill_wsu()
  
  p_octmar_exceedance <- df %>% group_by(group) %>% 
    mutate(precip = rollsum(precip, 7, align = "right", fill = NA)) %>% 
    group_by(group, water_year) %>%
    filter(month >= 10 | month <= 3) %>%
    mutate(prob = rank(precip)/(n()+1)) %>%
    group_by(group) %>%
    summarise(precip_80 = nth(precip, which.min(abs(prob-0.8))),
              precip_90 = nth(precip, which.min(abs(prob-0.9))),
              precip_95 = nth(precip, which.min(abs(prob-0.95)))) %>%
    ggplot() +
    geom_col(aes(x = group, y = precip_80, color = group, fill = group), alpha = 0.33) +
    geom_col(aes(x = group, y = precip_90, color = group, fill = group), alpha = 0.33) +
    geom_col(aes(x = group, y = precip_95, color = group, fill = group), alpha = 0.33) +
    # facet_grid(.~month_name, switch = "x") +
    labs(x = " ", y = " ") +
    theme_classic(base_size = size) +
    theme(axis.text.x=element_blank(),
          axis.ticks.x = element_blank(),
          legend.title = element_blank(),
          legend.position = "none") +
    scale_color_wsu() +
    scale_fill_wsu()
  
  p_exceedance_7 <- plot_grid(p_monthly_exceedance, p_yearly_exceedance, p_octmar_exceedance, rel_widths = c(6, 1, 1), align = 'h', nrow = 1, axis = "b")
  
  
  #### DRY DAYS PLOT ####
  p_monthly_dry <- df %>% group_by(month, year, group) %>%
    filter(precip == 0) %>% 
    summarise(dry_days = n(),
              month_name = month(time_stamp[1], label = TRUE)) %>%
    ggplot() +
    geom_boxplot(aes(x = group, y = dry_days, color = group, fill = group), alpha = 0.75) +
    # geom_jitter(aes(x = group, y = total_precip, color = group), alpha = 0.5, width = 0.05) +
    theme_classic(base_size = size) +
    facet_grid(.~month_name, switch = "x") +
    theme(axis.text.x = element_blank(),
          axis.ticks.x = element_blank(),
          legend.title = element_blank(),
          legend.position = "none",
          strip.background = element_blank(),
          strip.text = element_blank()) +
    scale_color_wsu() +
    scale_fill_wsu() +
    labs(y = "# Dry Days", x = " ")

  p_yearly_dry <- df %>% group_by(year, group) %>%
    filter(precip == 0) %>% 
    summarise(dry_days = n()) %>%
    ggplot() +
    geom_boxplot(aes(x = group, y = dry_days, color = group, fill = group), alpha = 0.75) +
    # geom_jitter(aes(x = group, y = total_precip, color = group), alpha = 0.5, width = 0.05) +
    theme_classic(base_size = size) +
    # facet_grid(.~month_name, switch = "x") +
    theme(axis.text.x = element_blank(),
          axis.ticks.x = element_blank(),
          legend.title = element_blank(),
          legend.position = "none") +
    scale_color_wsu() +
    scale_fill_wsu() +
    labs(y = " ", x = "Annual")

  p_octmar_dry <- df %>% mutate(water_year = year(time_stamp + month(3))) %>%
    group_by(group, water_year) %>%
    filter(month >= 10 | month <= 3, precip == 0) %>%
    summarise(dry_days = n()) %>%
    ggplot() +
    geom_boxplot(aes(x = group, y = dry_days, color = group, fill = group), alpha = 0.75) +
    # geom_jitter(aes(x = group, y = total_precip, color = group), alpha = 0.5, width = 0.05) +
    theme_classic(base_size = size) +
    # facet_grid(.~month_name, switch = "x") +
    theme(axis.text.x = element_blank(),
          axis.ticks.x = element_blank(),
          legend.title = element_blank(),
          legend.position = "none") +
    scale_color_wsu() +
    scale_fill_wsu() +
    labs(y = " ", x = "Oct - Mar")
  
  p_dry <- plot_grid(p_monthly_dry, p_yearly_dry, p_octmar_dry, rel_widths = c(6, 1, 1), align = 'h', nrow = 1, axis = "b")

  #### PRECIP PLOT ####
  p_monthly_precip <- df %>% group_by(month, year, group) %>%
    summarise(total_precip = sum(precip),
              month_name = month(time_stamp[1], label = TRUE)) %>%
    ggplot() +
    geom_boxplot(aes(x = group, y = total_precip, color = group, fill = group), alpha = 0.75) +
    # geom_jitter(aes(x = group, y = total_precip, color = group), alpha = 0.5, width = 0.05) +
    theme_classic(base_size = size) +
    facet_grid(.~month_name, switch = "x") +
    theme(axis.text.x = element_blank(),
          axis.ticks.x = element_blank(),
          legend.title = element_blank(),
          legend.position = "bottom",
          strip.background = element_blank()) +
    scale_color_wsu() +
    scale_fill_wsu() +
    labs(y = "Precip", x = " ")

  p_yearly_precip <- df %>% group_by(year, group) %>%
    summarise(total_precip = sum(precip)) %>%
    ggplot() +
    geom_boxplot(aes(x = group, y = total_precip, color = group, fill = group), alpha = 0.75) +
    # geom_jitter(aes(x = group, y = total_precip, color = group), alpha = 0.5, width = 0.05) +
    theme_classic(base_size = size) +
    # facet_grid(.~month_name, switch = "x") +
    theme(axis.text.x = element_blank(),
          axis.ticks.x = element_blank(),
          legend.title = element_blank(),
          legend.position = "none") +
    scale_color_wsu() +
    scale_fill_wsu() +
    labs(y = " ", x = "Annual")

  p_octmar_precip <- df %>% mutate(water_year = year(time_stamp + month(3))) %>%
    group_by(group, water_year) %>%
    filter(month >= 10 | month <= 3) %>%
    summarise(total_precip = sum(precip)) %>%
    ggplot() +
    geom_boxplot(aes(x = group, y = total_precip, color = group, fill = group), alpha = 0.75) +
    # geom_jitter(aes(x = group, y = total_precip, color = group), alpha = 0.5, width = 0.05) +
    theme_classic(base_size = size) +
    # facet_grid(.~month_name, switch = "x") +
    theme(axis.text.x = element_blank(),
          axis.ticks.x = element_blank(),
          legend.title = element_blank(),
          legend.position = "none") +
    scale_color_wsu() +
    scale_fill_wsu() +
    labs(y = " ", x = "Oct - Mar")

  #### FINAL PLOT COMBINE ####
  
  legend <- get_legend(p_monthly_precip)

  p_precip <- plot_grid(p_monthly_precip + theme(legend.position = "none"), p_yearly_precip, p_octmar_precip, rel_widths = c(6, 1, 1), align = 'h', nrow = 1, axis = "b")

  p_grid <- plot_grid(p_prob, p_prob_m, p_prob_7, p_exceedance, p_exceedance_m, p_exceedance_7, p_dry, p_precip, ncol = 1, align = "v", axis = 'l')

  plot_grid(p_grid, legend, ncol = 1, rel_heights = c(1,.05))
}
#******************************************************
#############
############# contour plots
#############
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
#******************************************************
#############
############# seasonal contour plots
#############

#******************************************************
#############
############# seasonal box plot
#############

#******************************************************
#############
############# 
#############

#******************************************************
#############
#############
#############
