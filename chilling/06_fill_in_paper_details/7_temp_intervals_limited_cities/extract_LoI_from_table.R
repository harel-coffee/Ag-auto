library(data.table)
library(dplyr)
options(digit=9)
options(digits=9)

data_dir <- "/Users/hn/Documents/00_GitHub/Ag_papers/Chill_Paper/tables/"
file <- "perc_table.csv"

perc_table <- read.csv(paste0(data_dir, file), as.is=TRUE)

wanted_cities <- c("Omak", "Walla Walla", "Yakima", "Eugene", "Richland")

perc_table <- perc_table %>% filter(city %in% wanted_cities) %>% data.table()

iof_char = c("(-Inf, -2]",
             "(-2, 4]",
             "(4, 6]",
             "(6, 8]",
             "(8, 13]",
             "(13, 16]",
             "(16, Inf]",
             "Four_to_13")

colnames(perc_table) <- c('city', "month", "time_period", iof_char)


cc <- c("sept_thru_dec_modeled", "sept_thru_jan_modeled")
perc_table <- perc_table %>% filter(month %in% cc) %>% data.table()

perc_table$month[perc_table$month == cc[1]] <- "Sept. - Dec."
perc_table$month[perc_table$month == cc[2]] <- "Sept. - Jan."

cols <- names(perc_table)[4:11]
perc_table[,(cols) := round(.SD,2), .SDcols=cols]

write.table(x = perc_table, row.names=F, col.names = T, sep=",", 
            file = paste0(data_dir, "perc_table_sept_DecJan_4_cities.csv"))