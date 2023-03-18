for (dir in list.dirs(path='/data/rajagopalan/CH_output_data/modeled')) {
  if (endsWith(dir, 'rcp85')) {
    print(dir)
    n <- 0
    for (path in list.files(path=dir, pattern="*.csv", full.names=T)) {
      if (!endsWith(path, 'consolidated.csv')) {
        df <- read.csv(path)
        new_df <- df[df$variety == 'Cabernet Sauvignon', ]
        if (nrow(new_df) < nrow(df)) {
          n <- n + 1
          write.csv(new_df, file=path, row.names = F)
        }
      }
    }
    print(n)
  }
}
