
rm(list = ls())
library(data.table)
library(dplyr)
library(lemon)
library(ggpubr)
library(purrr)

data_dir <- "/Users/hn/Documents/01_research_data/chilling/frost_bloom/"
source_dir <- "/Users/hn/Documents/00_GitHub/Ag/Bloom/"
param_dir <- paste0(source_dir, "parameters/")
plot_dir = "/Users/hn/Documents/00_GitHub/Ag_papers/Chill_Paper/figures_4_revision/panel/"
################################################################################################

limited_locations <- read.csv(file = paste0(param_dir, "limited_locations.csv"), as.is=TRUE)

limited_locations$location <- paste0(limited_locations$lat, "_", limited_locations$long)
limited_locations <- within(limited_locations, remove(lat, long))


daily_heat_for_CDF <- readRDS(paste0(data_dir, "/heat_accum_limit_cities.rds"))
daily_heat_for_CDF$location <- paste0(daily_heat_for_CDF$lat, "_", daily_heat_for_CDF$long)

daily_heat_for_CDF <- subset(daily_heat_for_CDF, 
                             select=c("year", "month", "day", "model", "emission", "location", "vert_Cum_dd"))

daily_heat_for_CDF <- dplyr::left_join(x = daily_heat_for_CDF, 
                                       y = limited_locations, 
                                       by = "location")

daily_heat_for_CDF <- daily_heat_for_CDF %>%
                      filter(city %in% c("Omak", "Walla Walla", "Eugene", "Yakima")) %>%
                      data.table()

CTs <- c("Omak", "Yakima", "Walla Walla", "Eugene")
daily_heat_for_CDF$city <- factor(daily_heat_for_CDF$city, levels = CTs, order=TRUE)

daily_heat_for_CDF$newDay = 1

daily_heat_for_CDF <- daily_heat_for_CDF %>% 
                      group_by(year, emission, city, model ) %>% 
                      mutate(DoY = cumsum(newDay)) %>% 
                      data.table()


daily_heat_for_CDF_observed <- daily_heat_for_CDF %>% 
                               filter(model == "observed") %>%
                               data.table()

# daily_heat_for_CDF_observed_45 <- daily_heat_for_CDF_observed
# daily_heat_for_CDF_observed_45$emission <- "RCP 4.5"
# daily_heat_for_CDF_observed <- rbind(daily_heat_for_CDF_observed, daily_heat_for_CDF_observed_45)

daily_heat_for_CDF_modeled <- daily_heat_for_CDF %>% 
                              filter(model != "observed" ) %>% 
                              data.table()

daily_heat_for_CDF_modeled_F1 <- daily_heat_for_CDF_modeled %>%
                                filter(year <= 2050) %>% 
                                data.table()

daily_heat_for_CDF_modeled_F2 <- daily_heat_for_CDF_modeled %>%
                               filter(year > 2050) %>% 
                               data.table()

daily_heat_for_CDF_modeled_F2 <- daily_heat_for_CDF_modeled_F2 %>%
                               filter(year <= 2075) %>% 
                               data.table()

daily_heat_for_CDF_modeled_F3 <- daily_heat_for_CDF_modeled %>%
                               filter(year > 2075) %>% 
                               data.table()

daily_heat_for_CDF_modeled_F1 <- rbind(daily_heat_for_CDF_modeled_F1, daily_heat_for_CDF_observed)
daily_heat_for_CDF_modeled_F2 <- rbind(daily_heat_for_CDF_modeled_F2, daily_heat_for_CDF_observed)
daily_heat_for_CDF_modeled_F3 <- rbind(daily_heat_for_CDF_modeled_F3, daily_heat_for_CDF_observed)


plot_Heat <- function(data, a_model, scenario_name, a_city){
  curr_dt <- data %>% 
             filter(city==a_city & emission == scenario_name & model==a_model)%>%
             data.table()

  the_theme <- theme(panel.grid.major = element_line(size=0.2),
                     panel.spacing=unit(.5, "cm"),
                   legend.text=element_text(size=24, face="plain"),
                   legend.title = element_blank(),
                   legend.position = "none",
                   strip.text = element_text(face="plain", size=14, color="black"),
                   axis.text = element_text(face="plain", size=12, color="black"),
                   axis.text.x = element_text(size=16, face="plain", 
                                              color="black", angle=-90),
                   axis.ticks = element_line(color = "black", size = .2),
                   axis.title.x = element_text(face="plain", size=15, 
                                               margin=margin(t=10, r=0, b=0, l=0)),
   
                   axis.title.y = element_text(face="plain", size=15, 
                                               margin=margin(t=0, r=10, b=0, l=0)),
   
                   plot.title = element_text(lineheight=.8, face="plain")
                   )

  ggplot(curr_dt, aes(x=DoY, y=vert_Cum_dd, colour = year)) +
        facet_grid(. ~ emission ~ city ~ model, scales="fixed") + 
        facet_rep_grid(. ~ emission ~ city ~ model, scales="fixed",  repeat.tick.labels = TRUE) + 
        geom_line(aes(color=factor(year) )) + 
        labs(x = "DoY", y = "cum. acc. heat", fill = "data type") +
        scale_x_continuous(breaks=c(  31,   59,    90,    120,    151,   181,  212,   243, 273,   304,   334, 365),
                           labels=c("J31", "F28", "M31", 'A30', "M31", "J30", "J31", "A31", "S30", "O31", "N30", "D31")
                           ) +  
        geom_vline(xintercept = 31, color = "black", size = 0.4) + 
        geom_vline(xintercept = 59, color = "black", size = 0.4) + 
        geom_vline(xintercept = 90, color = "black", size = 0.4) + 
        geom_vline(xintercept = 120, color = "black", size = 0.4) + 
        geom_vline(xintercept = 151, color = "black", size = 0.4) + 
        geom_vline(xintercept = 181, color = "black", size = 0.4) +
        geom_vline(xintercept = 212, color = "black", size = 0.4) +
        geom_vline(xintercept = 243, color = "black", size = 0.4) +
        the_theme
}

# a_model = unique(daily_heat_for_CDF_modeled_F1$model)[1]
# a_city = unique(daily_heat_for_CDF_modeled_F1$city)[1]
# a_scenario = unique(daily_heat_for_CDF_modeled_F1$emission)[1]

for(a_model in unique(daily_heat_for_CDF_modeled_F1$model)) {
  for(a_city in unique(daily_heat_for_CDF_modeled_F1$city)) {
    for(a_scenario in unique(daily_heat_for_CDF_modeled_F1$emission)) {
      sss = gsub(pattern = "\\.", replacement = "", x=a_scenario)
      sss = gsub(pattern = " ", replacement = "", x=sss)

      assign(x = paste(gsub(pattern = "-", replacement = "_", x = a_model), 
                       gsub(pattern = "-", replacement = "_", x = gsub(pattern = " ", replacement = "", x=a_city)), 
                       sss, 
                       sep="_"),
             value ={ plot_Heat(data=daily_heat_for_CDF_modeled_F1, 
                                a_model = a_model, 
                                scenario_name = a_scenario, 
                                a_city = a_city)}
            )
    }
  }
}


box_width = 25; box_height = 20

Yakima_f1_85 <- ggarrange(plotlist = list(observed_Yakima_RCP85,
                                          bcc_csm1_1_Yakima_RCP85,
                                          GFDL_ESM2G_Yakima_RCP85,
                                          HadGEM2_CC365_Yakima_RCP85,
                                          IPSL_CM5A_LR_Yakima_RCP85,
                                          inmcm4_Yakima_RCP85, 
                                          GFDL_ESM2M_Yakima_RCP85,
                                          HadGEM2_ES365_Yakima_RCP85,
                                          IPSL_CM5B_LR_Yakima_RCP85,
                                          MIROC5_Yakima_RCP85,
                                          MRI_CGCM3_Yakima_RCP85,
                                          NorESM1_M_Yakima_RCP85,
                                          MIROC_ESM_CHEM_Yakima_RCP85,
                                          IPSL_CM5A_MR_Yakima_RCP85,
                                          BNU_ESM_Yakima_RCP85,
                                          bcc_csm1_1_m_Yakima_RCP85,
                                          CanESM2_Yakima_RCP85,
                                          CCSM4_Yakima_RCP85,
                                          CNRM_CM5_Yakima_RCP85,
                                          CSIRO_Mk3_6_0_Yakima_RCP85),
                           ncol = 4, nrow = 5,
                           legend = "none",
                           common.legend = TRUE)


output_name = "Yakima_Heat_F1_85.png"
ggsave(output_name,
       Yakima_f1_85, 
       path=plot_dir, 
       width=box_width, height=box_height, unit="in", dpi=200,
       limitsize = FALSE)


Omak_f1_85 <- ggarrange(plotlist = list(observed_Omak_RCP85,
                                          bcc_csm1_1_Omak_RCP85,
                                          GFDL_ESM2G_Omak_RCP85,
                                          HadGEM2_CC365_Omak_RCP85,
                                          IPSL_CM5A_LR_Omak_RCP85,
                                          inmcm4_Omak_RCP85, 
                                          GFDL_ESM2M_Omak_RCP85,
                                          HadGEM2_ES365_Omak_RCP85,
                                          IPSL_CM5B_LR_Omak_RCP85,
                                          MIROC5_Omak_RCP85,
                                          MRI_CGCM3_Omak_RCP85,
                                          NorESM1_M_Omak_RCP85,
                                          MIROC_ESM_CHEM_Omak_RCP85,
                                          IPSL_CM5A_MR_Omak_RCP85,
                                          BNU_ESM_Omak_RCP85,
                                          bcc_csm1_1_m_Omak_RCP85,
                                          CanESM2_Omak_RCP85,
                                          CCSM4_Omak_RCP85,
                                          CNRM_CM5_Omak_RCP85,
                                          CSIRO_Mk3_6_0_Omak_RCP85),
                           ncol = 4, nrow = 5,
                           legend = "none",
                           common.legend = TRUE)


output_name = "Omak_Heat_F1_85.png"
ggsave(output_name,
       Omak_f1_85, 
       path=plot_dir, 
       width=box_width, height=box_height, unit="in", dpi=200,
       limitsize = FALSE)


WallaWalla_f1_85 <- ggarrange(plotlist = list(observed_WallaWalla_RCP85,
                                          bcc_csm1_1_WallaWalla_RCP85,
                                          GFDL_ESM2G_WallaWalla_RCP85,
                                          HadGEM2_CC365_WallaWalla_RCP85,
                                          IPSL_CM5A_LR_WallaWalla_RCP85,
                                          inmcm4_WallaWalla_RCP85, 
                                          GFDL_ESM2M_WallaWalla_RCP85,
                                          HadGEM2_ES365_WallaWalla_RCP85,
                                          IPSL_CM5B_LR_WallaWalla_RCP85,
                                          MIROC5_WallaWalla_RCP85,
                                          MRI_CGCM3_WallaWalla_RCP85,
                                          NorESM1_M_WallaWalla_RCP85,
                                          MIROC_ESM_CHEM_WallaWalla_RCP85,
                                          IPSL_CM5A_MR_WallaWalla_RCP85,
                                          BNU_ESM_WallaWalla_RCP85,
                                          bcc_csm1_1_m_WallaWalla_RCP85,
                                          CanESM2_WallaWalla_RCP85,
                                          CCSM4_WallaWalla_RCP85,
                                          CNRM_CM5_WallaWalla_RCP85,
                                          CSIRO_Mk3_6_0_WallaWalla_RCP85),
                           ncol = 4, nrow = 5,
                           legend = "none",
                           common.legend = TRUE)


output_name = "WallaWalla_Heat_F1_85.png"
ggsave(output_name,
       WallaWalla_f1_85, 
       path=plot_dir, 
       width=box_width, height=box_height, unit="in", dpi=200,
       limitsize = FALSE)

Eugene_f1_85 <- ggarrange(plotlist = list(observed_Eugene_RCP85,
                                          bcc_csm1_1_Eugene_RCP85,
                                          GFDL_ESM2G_Eugene_RCP85,
                                          HadGEM2_CC365_Eugene_RCP85,
                                          IPSL_CM5A_LR_Eugene_RCP85,
                                          inmcm4_Eugene_RCP85, 
                                          GFDL_ESM2M_Eugene_RCP85,
                                          HadGEM2_ES365_Eugene_RCP85,
                                          IPSL_CM5B_LR_Eugene_RCP85,
                                          MIROC5_Eugene_RCP85,
                                          MRI_CGCM3_Eugene_RCP85,
                                          NorESM1_M_Eugene_RCP85,
                                          MIROC_ESM_CHEM_Eugene_RCP85,
                                          IPSL_CM5A_MR_Eugene_RCP85,
                                          BNU_ESM_Eugene_RCP85,
                                          bcc_csm1_1_m_Eugene_RCP85,
                                          CanESM2_Eugene_RCP85,
                                          CCSM4_Eugene_RCP85,
                                          CNRM_CM5_Eugene_RCP85,
                                          CSIRO_Mk3_6_0_Eugene_RCP85),
                           ncol = 4, nrow = 5,
                           legend = "none",
                           common.legend = TRUE)


output_name = "Eugene_Heat_F1_85.png"
ggsave(output_name,
       Eugene_f1_85, 
       path=plot_dir, 
       width=box_width, height=box_height, unit="in", dpi=200,
       limitsize = FALSE)

################################
################################ F2
################################
for(a_model in unique(daily_heat_for_CDF_modeled_F2$model)) {
  for(a_city in unique(daily_heat_for_CDF_modeled_F2$city)) {
    for(a_scenario in unique(daily_heat_for_CDF_modeled_F2$emission)) {
      sss = gsub(pattern = "\\.", replacement = "", x=a_scenario)
      sss = gsub(pattern = " ", replacement = "", x=sss)

      assign(x = paste(gsub(pattern = "-", replacement = "_", x = a_model), 
                       gsub(pattern = "-", replacement = "_", x = gsub(pattern = " ", replacement = "", x=a_city)), 
                       sss, 
                       sep="_"),
             value ={ plot_Heat(data=daily_heat_for_CDF_modeled_F2, 
                                a_model = a_model, 
                                scenario_name = a_scenario, 
                                a_city = a_city)}
            )
    }
  }
}


Yakima_F2_85 <- ggarrange(plotlist = list(observed_Yakima_RCP85,
                                          bcc_csm1_1_Yakima_RCP85,
                                          GFDL_ESM2G_Yakima_RCP85,
                                          HadGEM2_CC365_Yakima_RCP85,
                                          IPSL_CM5A_LR_Yakima_RCP85,
                                          inmcm4_Yakima_RCP85, 
                                          GFDL_ESM2M_Yakima_RCP85,
                                          HadGEM2_ES365_Yakima_RCP85,
                                          IPSL_CM5B_LR_Yakima_RCP85,
                                          MIROC5_Yakima_RCP85,
                                          MRI_CGCM3_Yakima_RCP85,
                                          NorESM1_M_Yakima_RCP85,
                                          MIROC_ESM_CHEM_Yakima_RCP85,
                                          IPSL_CM5A_MR_Yakima_RCP85,
                                          BNU_ESM_Yakima_RCP85,
                                          bcc_csm1_1_m_Yakima_RCP85,
                                          CanESM2_Yakima_RCP85,
                                          CCSM4_Yakima_RCP85,
                                          CNRM_CM5_Yakima_RCP85,
                                          CSIRO_Mk3_6_0_Yakima_RCP85),
                           ncol = 4, nrow = 5,
                           legend = "none",
                           common.legend = TRUE)


output_name = "Yakima_Heat_F2_85.png"
ggsave(output_name,
       Yakima_F2_85, 
       path=plot_dir, 
       width=box_width, height=box_height, unit="in", dpi=200,
       limitsize = FALSE)


Omak_F2_85 <- ggarrange(plotlist = list(observed_Omak_RCP85,
                                          bcc_csm1_1_Omak_RCP85,
                                          GFDL_ESM2G_Omak_RCP85,
                                          HadGEM2_CC365_Omak_RCP85,
                                          IPSL_CM5A_LR_Omak_RCP85,
                                          inmcm4_Omak_RCP85, 
                                          GFDL_ESM2M_Omak_RCP85,
                                          HadGEM2_ES365_Omak_RCP85,
                                          IPSL_CM5B_LR_Omak_RCP85,
                                          MIROC5_Omak_RCP85,
                                          MRI_CGCM3_Omak_RCP85,
                                          NorESM1_M_Omak_RCP85,
                                          MIROC_ESM_CHEM_Omak_RCP85,
                                          IPSL_CM5A_MR_Omak_RCP85,
                                          BNU_ESM_Omak_RCP85,
                                          bcc_csm1_1_m_Omak_RCP85,
                                          CanESM2_Omak_RCP85,
                                          CCSM4_Omak_RCP85,
                                          CNRM_CM5_Omak_RCP85,
                                          CSIRO_Mk3_6_0_Omak_RCP85),
                           ncol = 4, nrow = 5,
                           legend = "none",
                           common.legend = TRUE)


output_name = "Omak_Heat_F2_85.png"
ggsave(output_name,
       Omak_F2_85, 
       path=plot_dir, 
       width=box_width, height=box_height, unit="in", dpi=200,
       limitsize = FALSE)


WallaWalla_F2_85 <- ggarrange(plotlist = list(observed_WallaWalla_RCP85,
                                          bcc_csm1_1_WallaWalla_RCP85,
                                          GFDL_ESM2G_WallaWalla_RCP85,
                                          HadGEM2_CC365_WallaWalla_RCP85,
                                          IPSL_CM5A_LR_WallaWalla_RCP85,
                                          inmcm4_WallaWalla_RCP85, 
                                          GFDL_ESM2M_WallaWalla_RCP85,
                                          HadGEM2_ES365_WallaWalla_RCP85,
                                          IPSL_CM5B_LR_WallaWalla_RCP85,
                                          MIROC5_WallaWalla_RCP85,
                                          MRI_CGCM3_WallaWalla_RCP85,
                                          NorESM1_M_WallaWalla_RCP85,
                                          MIROC_ESM_CHEM_WallaWalla_RCP85,
                                          IPSL_CM5A_MR_WallaWalla_RCP85,
                                          BNU_ESM_WallaWalla_RCP85,
                                          bcc_csm1_1_m_WallaWalla_RCP85,
                                          CanESM2_WallaWalla_RCP85,
                                          CCSM4_WallaWalla_RCP85,
                                          CNRM_CM5_WallaWalla_RCP85,
                                          CSIRO_Mk3_6_0_WallaWalla_RCP85),
                           ncol = 4, nrow = 5,
                           legend = "none",
                           common.legend = TRUE)


output_name = "WallaWalla_Heat_F2_85.png"
ggsave(output_name,
       WallaWalla_F2_85, 
       path=plot_dir, 
       width=box_width, height=box_height, unit="in", dpi=200,
       limitsize = FALSE)

Eugene_F2_85 <- ggarrange(plotlist = list(observed_Eugene_RCP85,
                                          bcc_csm1_1_Eugene_RCP85,
                                          GFDL_ESM2G_Eugene_RCP85,
                                          HadGEM2_CC365_Eugene_RCP85,
                                          IPSL_CM5A_LR_Eugene_RCP85,
                                          inmcm4_Eugene_RCP85, 
                                          GFDL_ESM2M_Eugene_RCP85,
                                          HadGEM2_ES365_Eugene_RCP85,
                                          IPSL_CM5B_LR_Eugene_RCP85,
                                          MIROC5_Eugene_RCP85,
                                          MRI_CGCM3_Eugene_RCP85,
                                          NorESM1_M_Eugene_RCP85,
                                          MIROC_ESM_CHEM_Eugene_RCP85,
                                          IPSL_CM5A_MR_Eugene_RCP85,
                                          BNU_ESM_Eugene_RCP85,
                                          bcc_csm1_1_m_Eugene_RCP85,
                                          CanESM2_Eugene_RCP85,
                                          CCSM4_Eugene_RCP85,
                                          CNRM_CM5_Eugene_RCP85,
                                          CSIRO_Mk3_6_0_Eugene_RCP85),
                           ncol = 4, nrow = 5,
                           legend = "none",
                           common.legend = TRUE)


output_name = "Eugene_Heat_F2_85.png"
ggsave(output_name,
       Eugene_F2_85, 
       path=plot_dir, 
       width=box_width, height=box_height, unit="in", dpi=200,
       limitsize = FALSE)


################################
################################ F3
################################
for(a_model in unique(daily_heat_for_CDF_modeled_F3$model)) {
  for(a_city in unique(daily_heat_for_CDF_modeled_F3$city)) {
    for(a_scenario in unique(daily_heat_for_CDF_modeled_F3$emission)) {
      sss = gsub(pattern = "\\.", replacement = "", x=a_scenario)
      sss = gsub(pattern = " ", replacement = "", x=sss)

      assign(x = paste(gsub(pattern = "-", replacement = "_", x = a_model), 
                       gsub(pattern = "-", replacement = "_", x = gsub(pattern = " ", replacement = "", x=a_city)), 
                       sss, 
                       sep="_"),
             value ={ plot_Heat(data=daily_heat_for_CDF_modeled_F3, 
                                a_model = a_model, 
                                scenario_name = a_scenario, 
                                a_city = a_city)}
            )
    }
  }
}


Yakima_F3_85 <- ggarrange(plotlist = list(observed_Yakima_RCP85,
                                          bcc_csm1_1_Yakima_RCP85,
                                          GFDL_ESM2G_Yakima_RCP85,
                                          HadGEM2_CC365_Yakima_RCP85,
                                          IPSL_CM5A_LR_Yakima_RCP85,
                                          inmcm4_Yakima_RCP85, 
                                          GFDL_ESM2M_Yakima_RCP85,
                                          HadGEM2_ES365_Yakima_RCP85,
                                          IPSL_CM5B_LR_Yakima_RCP85,
                                          MIROC5_Yakima_RCP85,
                                          MRI_CGCM3_Yakima_RCP85,
                                          NorESM1_M_Yakima_RCP85,
                                          MIROC_ESM_CHEM_Yakima_RCP85,
                                          IPSL_CM5A_MR_Yakima_RCP85,
                                          BNU_ESM_Yakima_RCP85,
                                          bcc_csm1_1_m_Yakima_RCP85,
                                          CanESM2_Yakima_RCP85,
                                          CCSM4_Yakima_RCP85,
                                          CNRM_CM5_Yakima_RCP85,
                                          CSIRO_Mk3_6_0_Yakima_RCP85),
                           ncol = 4, nrow = 5,
                           legend = "none",
                           common.legend = TRUE)


output_name = "Yakima_Heat_F3_85.png"
ggsave(output_name,
       Yakima_F3_85, 
       path=plot_dir, 
       width=box_width, height=box_height, unit="in", dpi=200,
       limitsize = FALSE)


Omak_F3_85 <- ggarrange(plotlist = list(observed_Omak_RCP85,
                                          bcc_csm1_1_Omak_RCP85,
                                          GFDL_ESM2G_Omak_RCP85,
                                          HadGEM2_CC365_Omak_RCP85,
                                          IPSL_CM5A_LR_Omak_RCP85,
                                          inmcm4_Omak_RCP85, 
                                          GFDL_ESM2M_Omak_RCP85,
                                          HadGEM2_ES365_Omak_RCP85,
                                          IPSL_CM5B_LR_Omak_RCP85,
                                          MIROC5_Omak_RCP85,
                                          MRI_CGCM3_Omak_RCP85,
                                          NorESM1_M_Omak_RCP85,
                                          MIROC_ESM_CHEM_Omak_RCP85,
                                          IPSL_CM5A_MR_Omak_RCP85,
                                          BNU_ESM_Omak_RCP85,
                                          bcc_csm1_1_m_Omak_RCP85,
                                          CanESM2_Omak_RCP85,
                                          CCSM4_Omak_RCP85,
                                          CNRM_CM5_Omak_RCP85,
                                          CSIRO_Mk3_6_0_Omak_RCP85),
                           ncol = 4, nrow = 5,
                           legend = "none",
                           common.legend = TRUE)


output_name = "Omak_Heat_F3_85.png"
ggsave(output_name,
       Omak_F3_85, 
       path=plot_dir, 
       width=box_width, height=box_height, unit="in", dpi=200,
       limitsize = FALSE)


WallaWalla_F3_85 <- ggarrange(plotlist = list(observed_WallaWalla_RCP85,
                                          bcc_csm1_1_WallaWalla_RCP85,
                                          GFDL_ESM2G_WallaWalla_RCP85,
                                          HadGEM2_CC365_WallaWalla_RCP85,
                                          IPSL_CM5A_LR_WallaWalla_RCP85,
                                          inmcm4_WallaWalla_RCP85, 
                                          GFDL_ESM2M_WallaWalla_RCP85,
                                          HadGEM2_ES365_WallaWalla_RCP85,
                                          IPSL_CM5B_LR_WallaWalla_RCP85,
                                          MIROC5_WallaWalla_RCP85,
                                          MRI_CGCM3_WallaWalla_RCP85,
                                          NorESM1_M_WallaWalla_RCP85,
                                          MIROC_ESM_CHEM_WallaWalla_RCP85,
                                          IPSL_CM5A_MR_WallaWalla_RCP85,
                                          BNU_ESM_WallaWalla_RCP85,
                                          bcc_csm1_1_m_WallaWalla_RCP85,
                                          CanESM2_WallaWalla_RCP85,
                                          CCSM4_WallaWalla_RCP85,
                                          CNRM_CM5_WallaWalla_RCP85,
                                          CSIRO_Mk3_6_0_WallaWalla_RCP85),
                           ncol = 4, nrow = 5,
                           legend = "none",
                           common.legend = TRUE)


output_name = "WallaWalla_Heat_F3_85.png"
ggsave(output_name,
       WallaWalla_F3_85, 
       path=plot_dir, 
       width=box_width, height=box_height, unit="in", dpi=200,
       limitsize = FALSE)

Eugene_F3_85 <- ggarrange(plotlist = list(observed_Eugene_RCP85,
                                          bcc_csm1_1_Eugene_RCP85,
                                          GFDL_ESM2G_Eugene_RCP85,
                                          HadGEM2_CC365_Eugene_RCP85,
                                          IPSL_CM5A_LR_Eugene_RCP85,
                                          inmcm4_Eugene_RCP85, 
                                          GFDL_ESM2M_Eugene_RCP85,
                                          HadGEM2_ES365_Eugene_RCP85,
                                          IPSL_CM5B_LR_Eugene_RCP85,
                                          MIROC5_Eugene_RCP85,
                                          MRI_CGCM3_Eugene_RCP85,
                                          NorESM1_M_Eugene_RCP85,
                                          MIROC_ESM_CHEM_Eugene_RCP85,
                                          IPSL_CM5A_MR_Eugene_RCP85,
                                          BNU_ESM_Eugene_RCP85,
                                          bcc_csm1_1_m_Eugene_RCP85,
                                          CanESM2_Eugene_RCP85,
                                          CCSM4_Eugene_RCP85,
                                          CNRM_CM5_Eugene_RCP85,
                                          CSIRO_Mk3_6_0_Eugene_RCP85),
                           ncol = 4, nrow = 5,
                           legend = "none",
                           common.legend = TRUE)


output_name = "Eugene_Heat_F3_85.png"
ggsave(output_name,
       Eugene_F3_85, 
       path=plot_dir, 
       width=box_width, height=box_height, unit="in", dpi=200,
       limitsize = FALSE)


