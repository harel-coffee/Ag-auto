# read files off the disk


historical_txt_data_path <- "/data/hydro/users/giridhar/giridhar/codmoth_pop/alldata_us_locations/data_processed/historical"
path_for_RDS <- "/home/hnoorazar/data/historical_data_RDS"


#######
#######  Convert ASCII to RDS.
#######
read_ascii_write_rds = function(path_to_assci_files, path_to_rds_files){
  # list of all files (that can be seen, i.e. not hidden files) in the path_to_assci_files
  file_names_list_to_read <- list.files(path_to_assci_files, full.names = F)
  
  for (file in file_names_list_to_read){
    # attach path to file names.
    input_file_full_name <- paste(path_to_assci_files, "/", file, sep="")
    
    # read a given file
    current_file <- read.table(input_file_full_name)
    
    # full path-file name to be written
    output_file_full_name <- paste(path_to_rds_files, "/", file, ".rds", sep="")
    
    # save the file in RDS format.
    saveRDS(current_file, output_file_full_name)
  }
}
####################







