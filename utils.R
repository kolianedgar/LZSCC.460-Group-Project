set_script_wd <- function() {
  if (!is.null(sys.frames()[[1]]$ofile)) {
    # source()
    return(setwd(dirname(normalizePath(sys.frames()[[1]]$ofile))))
  }
  
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- "--file="
  if (any(grepl(file_arg, args))) {
    script_path <- sub(file_arg, "", args[grep(file_arg, args)])
    return(setwd(dirname(normalizePath(script_path))))
  }
  
  # fallback (do nothing)
  invisible(NULL)
}