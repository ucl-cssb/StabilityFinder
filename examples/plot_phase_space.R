library(ggplot2)
library(gridExtra)
library(grid)
plot_stabilityChecker_particles <- function(numb_files, filename){
  filelist <- list.files(pattern = "set_result*")
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

  pdf('phase_plots.pdf')
  do.call("grid.arrange", pltList)
  dev.off()


}
plot_stabilityChecker_particles(10, "switch_result")



