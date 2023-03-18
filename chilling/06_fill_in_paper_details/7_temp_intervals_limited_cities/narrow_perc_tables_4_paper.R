


data_dir <- "/Users/hn/Documents/01_research_data/chilling/7_temp_int_limit_locs/untitled/RDS_files/sept1_Dec_31_tables/"

perc_tbl <- read.csv(paste0(data_dir, "perc_table.csv"))
colnames(perc_tbl) <- c("city", "month", "time_period", "(-Inf, -2]", "(-2, 4]", "(4, 6]", "(6, 8]", "(8, 13]", "(13, 16]", "(16, Inf]", "Four_to_13")

perc_tbl_2 <- perc_tbl %>% filter(city %in% c("Omak", "Yakima", "Walla Walla", "Eugene"))
write.table(x = perc_tbl_2, row.names=F, col.names = T, sep=",", file = paste0(data_dir, "/perc_table_4_cities.csv"))

perc_tbl_2['-2 C - 13 C'] <- perc_tbl_2['(-2, 4]'] + perc_tbl_2['(4, 6]'] + perc_tbl_2["(6, 8]"]+ perc_tbl_2["(8, 13]"]
perc_tbl_2['> 13 C'] <- perc_tbl_2["(13, 16]"] +  perc_tbl_2["(16, Inf]"]

keeps <- c("city", "month","time_period", "(-Inf, -2]", "-2 C - 13 C", "> 13 C")
perc_tbl_2 <- perc_tbl_2[ , keeps, drop = FALSE]

write.table(x = perc_tbl_2, row.names=F, col.names = T, sep=",", file = paste0(data_dir, "/perc_table_4_cities_4_paper_narrowed.csv"))