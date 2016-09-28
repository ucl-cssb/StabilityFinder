args = commandArgs(trailingOnly=TRUE)


# test if there is at least one argument: if not, return an error
if (length(args)==0) {
  stop("At least one argument must be supplied (results files directory).n", call.=FALSE)
} else if (length(args)==1) {
  # default output file
  args[2] = "phase_plots.pdf"
}


library(ggplot2)
library(gridExtra)
library(grid)
plot_stabilityChecker_particles <- function(){
  setwd(args[1])
  filelist <- list.files(pattern = "set_result*")
  numb_files <- length(filelist)
  data_list = lapply(filelist, read.table, sep = " ")
  pltList <- list()
  for(i in 1:numb_files){
    pltList[[i]] <-ggplot(data_list[[i]], aes(x=V1, y=V2)) +
      #xlim(0,10)+
      #ylim(0,10)+
      #stat_density2d(aes(alpha=..level.., fill=..level..),
      #               size=2, bins=50, geom="polygon") +
      #scale_fill_gradient(low = "yellow", high = "red") +
      #scale_alpha(range = c(0.00, 0.5), guide = FALSE) +
      geom_point()+
      #geom_density2d(colour="black", bins=10)+
      theme(axis.line=element_blank(),
            axis.title.x=element_blank(),
            axis.title.y=element_blank(),
            legend.position="none",
            panel.background=element_blank(),
            panel.border=element_blank(),
            panel.grid.minor=element_blank(),
            plot.background=element_blank())+
    theme(axis.text.x = element_text(angle = 60, hjust = 1, vjust = 1))
  }

  pdf(args[2])
  do.call("grid.arrange", pltList)
  dev.off()


}
plot_stabilityChecker_particles()



