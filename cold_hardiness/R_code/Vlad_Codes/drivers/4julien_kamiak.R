names <- c('Walla Walla', 'Richland', 'Wenatchee', 'Omak')
lats <- c(46.03125, 46.28125, 47.40625, 48.40625)
longs <- c(-118.34375, -119.34375, -120.34375, -119.53125)

for (i in 1:length(names)) {
  for (dir in list.dirs('/data/adam/data/metdata/maca_v2_vic_binary', recursive=F)) {
    for (rcp in c('rcp45', 'rcp85')) {
      filename <- sprintf('%s/%s/data_%s_%s', dir, rcp, lats[i], longs[i])
      if (file.exists(filename)) {
        print(filename)
        file.copy(filename, sprintf('/data/rajagopalan/CH_output_data/4julien_binary/%s_%s_%s_%s_%s',
                                    names[i], basename(dir), rcp, lats[i], longs[i]))
      }
    }
  }
}