###################################################################
#          **********                            **********
#          **********        WARNING !!!!        **********
#          **********                            **********
##
## DO NOT load any libraries here.
## And do not load any libraries on the drivers!
## Unless you are aware of conflicts between packages.
## I spent hours to figrue out what the hell is going on!
###################################################################

.libPaths("/data/hydro/R_libs35")
.libPaths()


######################################################################
##                                                                  ##
##              Terminal/shell/bash arguments                       ##
##                                                                  ##
######################################################################
start_time <- Sys.time()

cold_out = "/data/hydro/users/Hossein/cold_hardiness/"

if (dir.exists(cold_out) == F) {dir.create(path = cold_out, recursive = T)}

dir_con <- dir()
dir_con <- dir_con[grep(pattern = "data_",
                        x = dir_con)]

dir_con <- dir_con[which(dir_con %in% local_files)]

print(dir_con)


end_time <- Sys.time()
print( end_time - start_time)

