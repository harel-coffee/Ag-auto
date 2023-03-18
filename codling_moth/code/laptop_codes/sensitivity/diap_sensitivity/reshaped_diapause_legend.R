rm(list=ls())
library(ggplot2)

dp_1 = c(102.6077, 1.306483, 16.95815) 
dp_2 = c(103, 1.1, 16.8) 
dp_3 = c(103, 0.95, 16.5)
dp_4 = c(104, 0.8, 16.3) 
dp_5 = c(103, 0.8, 15.95)
dp_6 = c(104, 0.7, 15.6) 
dp_7 = c(104, 0.6, 15.3) 

diapause_1 <- function(x){dp_1[1] * exp(-exp(dp_1[2] * (x - dp_1[3] ))) } # original
diapause_2 <- function(x){dp_2[1] * exp(-exp(dp_2[2] * (x - dp_2[3] ))) }
diapause_3 <- function(x){dp_3[1] * exp(-exp(dp_3[2] * (x - dp_3[3] ))) }
diapause_4 <- function(x){dp_4[1] * exp(-exp(dp_4[2] * (x - dp_4[3] ))) }
diapause_5 <- function(x){dp_5[1] * exp(-exp(dp_5[2] * (x - dp_5[3] ))) }
diapause_6 <- function(x){dp_6[1] * exp(-exp(dp_6[2] * (x - dp_6[3] ))) }
diapause_7 <- function(x){dp_7[1] * exp(-exp(dp_7[2] * (x - dp_7[3] ))) }

diapause_reshaped_plot = ggplot(data = data.frame(x = 0), 
                                mapping = aes(x = x)) + 
                         theme_bw()+
                         labs(x="Day length hours", y="% larva diapaused induced") + 
                         xlim(5, 20) + 
                         stat_function(fun = diapause_1, aes(colour = "original", linetype="original")) +
                         stat_function(fun = diapause_2, aes(colour = "param. 2", linetype="param. 2")) + 
                         stat_function(fun = diapause_3, aes(colour = "param. 3", linetype="param. 3")) +
                         stat_function(fun = diapause_4, aes(colour = "param. 4", linetype="param. 4")) +
                         stat_function(fun = diapause_5, aes(colour = "param. 5", linetype="param. 5")) + 
                         stat_function(fun = diapause_6, aes(colour = "param. 6", linetype="param. 6")) + 
                         stat_function(fun = diapause_7, aes(colour = "param. 7", linetype="param. 7")) + 
                         # scale_colour_manual("", values = c("black", "red", "blue", "green", "black", "red", "orange"))
                         scale_linetype_manual(values = c("solid", "solid", "solid", "solid", "dashed", "dashed", "solid"), guide = FALSE) +
                         scale_colour_manual(values = c("black", "red", "blue", "green", "black", "red", "orange"),
                                            guide = guide_legend(override.aes = list(
                                            linetype = c("solid", "solid", "solid", "solid", "dashed", "dashed", "solid"),
                                            size = c(.1, .1, .1, .1, .1, .1, .1)),
                                            title = NULL))

diapause_reshaped_plot

plot_name = paste0("diapause_reshaped_plot.png")
plot_path = "/Users/hn/Documents/GitHub/Kirti/codling_moth/to_write_paper/figures/sensitivity/Diapause/"
ggsave(plot_name, diapause_reshaped_plot, 
       device="png", path=plot_path, 
       width=10, height=7, unit="in", dpi=300)


rm(list=ls())
library(ggplot2)
dp_1 = c(102.6077, 1.306483, 16.95815) 
dp_2 = c(103, 1.1, 16.8) 
dp_3 = c(103, 0.95, 16.5)
dp_4 = c(104, 0.8, 16.3) 
dp_5 = c(103, 0.8, 15.95)
dp_6 = c(104, 0.7, 15.6) 
dp_7 = c(104, 0.6, 15.3) 

# original
diapause_1 <- function(x){
  ifelse(dp_1[1] * exp(-exp(dp_1[2] * (x - dp_1[3]))) > 100, 100, dp_1[1] * exp(-exp(dp_1[2] * (x - dp_1[3]))))
}

diapause_2 <- function(x){
  ifelse(dp_2[1] * exp(-exp(dp_2[2] * (x - dp_2[3])))>100, 100, dp_2[1] * exp(-exp(dp_2[2] * (x - dp_2[3]))))
}

diapause_3 <- function(x){
  ifelse(dp_3[1] * exp(-exp(dp_3[2] * (x - dp_3[3] ))) > 100, 100, dp_3[1] * exp(-exp(dp_3[2] * (x - dp_3[3] ))))
}

diapause_4 <- function(x){
  ifelse(dp_4[1] * exp(-exp(dp_4[2] * (x - dp_4[3] ))) > 100, 100, dp_4[1] * exp(-exp(dp_4[2] * (x - dp_4[3] ))))
}

diapause_5 <- function(x){
  ifelse(dp_5[1] * exp(-exp(dp_5[2] * (x - dp_5[3] ))) > 100, 100, dp_5[1] * exp(-exp(dp_5[2] * (x - dp_5[3] ))))
}

diapause_6 <- function(x){
  ifelse(dp_6[1] * exp(-exp(dp_6[2] * (x - dp_6[3] ))) > 100, 100, dp_6[1] * exp(-exp(dp_6[2] * (x - dp_6[3] ))))
}

# orange
diapause_7 <- function(x){
  ifelse(dp_7[1] * exp(-exp(dp_7[2] * (x - dp_7[3] ))) > 100, 100, dp_7[1] * exp(-exp(dp_7[2] * (x - dp_7[3] ))))
}

diapause_reshaped_plot = ggplot(data = data.frame(x = 0), 
                                mapping = aes(x = x)) + 
                         theme_bw()+
                         labs(x="Day length hours", y="% larva diapaused induced") + 
                         xlim(5, 20) + 
                         stat_function(fun = diapause_1, aes(colour = "original", linetype="original")) +
                         stat_function(fun = diapause_2, aes(colour = "param. 2", linetype="param. 2")) + 
                         stat_function(fun = diapause_3, aes(colour = "param. 3", linetype="param. 3")) +
                         stat_function(fun = diapause_4, aes(colour = "param. 4", linetype="param. 4")) +
                         stat_function(fun = diapause_5, aes(colour = "param. 5", linetype="param. 5")) + 
                         stat_function(fun = diapause_6, aes(colour = "param. 6", linetype="param. 6")) + 
                         stat_function(fun = diapause_7, aes(colour = "param. 7", linetype="param. 7")) + 
                         # scale_colour_manual("", values = c("black", "red", "blue", "green", "black", "red", "orange"))
                         scale_linetype_manual(values = c("solid", "solid", "solid", "solid", "dashed", "dashed", "solid"), guide = FALSE) +
                         scale_colour_manual(values = c("black", "red", "blue", "green", "black", "red", "orange"),
                                            guide = guide_legend(override.aes = list(
                                            linetype = c("solid", "solid", "solid", "solid", "dashed", "dashed", "solid"),
                                            size = c(.1, .1, .1, .1, .1, .1, .1)),
                                            title = NULL))

diapause_reshaped_plot

plot_name = paste0("diap_reshaped_plot.png")
plot_path = "/Users/hn/Documents/GitHub/Kirti/codling_moth/to_write_paper/figures/sensitivity/Diapause/"
ggsave(plot_name, diapause_reshaped_plot, 
       device="png", path=plot_path, 
       width=10, height=7, unit="in", dpi=400)

