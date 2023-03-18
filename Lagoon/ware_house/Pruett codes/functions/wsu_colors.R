wsu_colors <- c(
  `red`        = "#c60c30",
  `green`      = "#ada400",
  `blue`       = "#00a5bd",
  `orange`     = "#f6861f",
  `yellow`     = "#ffb81c",
  `light grey` = "#8d959a",
  `dark grey`  = "#5e6a71")

wsu_cols <- function(...) {
  cols <- c(...)
  
  if (is.null(cols))
    return (wsu_colors)
  
  wsu_colors[cols]
}

wsu_palettes <- list(
  `main`  = wsu_cols("red", "blue", "green", "yellow"),
  
  `rev`  = wsu_cols("blue", "green", "yellow", "red"),
  
  `mo_red`  = wsu_cols("blue", "green", "yellow"),
  
  `cool`  = wsu_cols("blue", "green"),
  
  `hot`   = wsu_cols("yellow", "orange", "red"),
  
  `mixed` = wsu_cols("blue", "green", "yellow", "orange", "red"),
  
  `grey`  = wsu_cols("light grey", "dark grey")
)

wsu_pal <- function(palette = "main", reverse = FALSE, ...) {
  pal <- wsu_palettes[[palette]]
  
  if (reverse) pal <- rev(pal)
  
  colorRampPalette(pal, ...)
}

scale_color_wsu <- function(palette = "main", discrete = TRUE, reverse = FALSE, ...) {
  pal <- wsu_pal(palette = palette, reverse = reverse)
  
  if (discrete) {
    discrete_scale("colour", paste0("wsu_", palette), palette = pal, ...)
  } else {
    scale_color_gradientn(colours = pal(256), ...)
  }
}

scale_fill_wsu <- function(palette = "main", discrete = TRUE, reverse = FALSE, ...) {
  pal <- wsu_pal(palette = palette, reverse = reverse)
  
  if (discrete) {
    discrete_scale("fill", paste0("wsu_", palette), palette = pal, ...)
  } else {
    scale_fill_gradientn(colours = pal(256), ...)
  }
}


