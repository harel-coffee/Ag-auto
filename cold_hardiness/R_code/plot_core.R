plot_hardiness<-function(data, plot_dir, out_name, width=8, height=5){
	output_plot  <- subset(data, select=c(jday, t_max, t_min, 
	                                      predicted_Hc, observed_Hc, 
	                                      budbreak))
	output_plot <- melt(output_plot, id=c("jday"))


	budbreak_date = data[which(! is.na(data$budbreak))]$Date

	hard_plot<- ggplot(output_plot, aes(x=jday, y=value, shape=variable, color=variable)) +
				labs(x = "Julian day", y = "Temperature", 
					 title=paste0(data$variety[1], " ", data$location[1], " ",
					 	          data$year[1], "\n", "Budbreak predicted on ", 
					 	          budbreak_date)) +
				geom_point() +
				geom_path() + 
				theme_bw() +
				theme(plot.title = element_text(color="black", size=14, 
					                            face="bold", family="Times", 
					                            hjust = 0.5),
					  legend.position="right",
					  legend.title=element_blank(),
					  legend.text = element_text(size=10)
					  ) +
				# to remove duplicate legend text
				guides(guide_legend(title=element_blank()), shape = FALSE) + 
				scale_color_discrete(#name="Experimental\nCondition",
				                      breaks=c("t_max", "t_min", "predicted_Hc", 
				                      	       "observed_Hc", "budbreak"),
				                      labels=c("Max. temp.", "Min. temp.", 
				                               "Predicted hard.", "Observed hard.", 
				                               "Budbreak"))
	out_name = paste0(out_name, ".png")
    ggsave(out_name, hard_plot, device="png", 
    	   path=plot_dir, width=width, height=height, 
           unit="in", dpi=500)
}

