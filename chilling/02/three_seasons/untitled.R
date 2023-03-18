file <- read.table(file = "/Users/hn/Documents/GitHub/Ag/chilling/02/three_seasons/chill_output_data_48.96875_-122.65625_future.txt",
                   header = T, 
                   colClasses = c("factor", "numeric", "numeric", "numeric",
                                  "numeric", "numeric"))

data_list_F[[i]] <- process_data_non_overlap(file, time_period="2005_2024")


file <- read.table(file = "/Users/hn/Documents/GitHub/Ag/chilling/02/three_seasons/chill_output_data_48.96875_-122.65625.txt", 
                   header = T, 
                   colClasses = c("factor", "numeric", "numeric", "numeric",
                                  "numeric", "numeric"))
names(data_list_historical)[1] <- "/Users/hn/Documents/GitHub/Ag/chilling/02/three_seasons/chill_output_data_48.96875_-122.65625.txt"