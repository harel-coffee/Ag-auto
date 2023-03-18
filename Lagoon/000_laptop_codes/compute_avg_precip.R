

ann_all_last_days <- readRDS("/Users/hn/Desktop/Desktop/Kirti/check_point/lagoon/precip/ann_all_last_days.rds")

head(ann_all_last_days, 2)
ann_all_last_days <- ann_all_last_days %>% filter(time_period == "1979-2016")

ann_all_last_days <- within(ann_all_last_days, remove(cluster, emission))
ann_all_last_days <- unique(ann_all_last_days)
ann_all_last_days <- subset(ann_all_last_days, select=c(location, annual_cum_precip))
ann_all_last_days <- data.table(ann_all_last_days)

ann_all_last_days %>%
group_by(location)%>%
summarise_at(vars(annual_cum_precip), funs(mean(., na.rm=TRUE)))

B <- ann_all_last_days[, .(precip_avg = mean(annual_cum_precip)), 
                         by = c("location")]

B <- ann_all_last_days[, .(precip_avg = mean(annual_cum_precip)), by = c("location")]
cols <- "precip_avg"
B[,(cols) := round(.SD, 2), .SDcols=cols]
head(B, 2)

saveRDS(B, "/Users/hn/Desktop/Desktop/Kirti/check_point/lagoon/precip_avgs.rds")


