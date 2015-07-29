library(ggplot2)
library(gridExtra)
library(XML)
library(plyr)
library(grid)

#Multiplot from R-cookbook.
multiplot <- function(..., plotlist=NULL, file, cols=1, layout=NULL) {
  library(grid)
  plots <- c(list(...), plotlist)

  numPlots = length(plots)

  if (is.null(layout)) {
    # Make the panel
    # ncol: Number of columns of plots
    # nrow: Number of rows needed, calculated from # of cols
    layout <- matrix(seq(1, cols * ceiling(numPlots/cols)),
                    ncol = cols, nrow = ceiling(numPlots/cols))
  }
 if (numPlots==1) {
    print(plots[[1]])

  } else {
    grid.newpage()
    pushViewport(viewport(layout = grid.layout(nrow(layout), ncol(layout))))
    for (i in 1:numPlots) {
      # Get the i,j matrix positions of the regions that contain this subplot
      matchidx <- as.data.frame(which(layout == i, arr.ind = TRUE))
      print(plots[[i]], vp = viewport(layout.pos.row = matchidx$row,
                                      layout.pos.col = matchidx$col))
    }
  }
}

plot_posterior_distr <- function(limits, param_names, p_values_final){

  numb_params = length(param_names)-1

  a=limits[,1]
  b=limits[,2]
  pltList <- list()
  k=0
  for(i in 1:numb_params)
    for(j in 1:numb_params){
      k=k+1
      if(i==j){
        pltList[[k]] <-ggplot(p_values_final, aes_string(x=param_names[i], weight=param_names[ncol(p_values_final)])) + geom_density(fill="grey") + xlim(a[i],b[i])+ ggtitle(param_names[i]) +
          theme(axis.line=element_blank(),
                plot.title=element_text(size=8, hjust=0,lineheight=0),
                axis.text.x=element_text(size=6,angle = 90, vjust=0,hjust=1.2),
                axis.text.y=element_text(size=6),
                axis.ticks=element_blank(),
                axis.title.x=element_blank(),
                axis.title.y=element_blank(),
                legend.position="none",
                panel.grid.minor=element_blank(),
                plot.background=element_blank(),
                plot.margin=unit(c(0,0,-0.5,0), "lines"))
      }else{
        pltList[[k]] <-ggplot(p_values_final, aes_string(x = param_names[i], y = param_names[j], weight=param_names[ncol(p_values_final)])) + xlim(a[i],b[i])+ ylim(a[j],b[j])+
          stat_density2d(aes(alpha=..level.., fill=..level.., weight=weights),
                         size=2, bins=10, geom="polygon") +
          scale_fill_gradient(low = "yellow", high = "red") +
          scale_alpha(range = c(0.00, 0.5), guide = FALSE) +
          geom_density2d(colour="black", bins=10)+
          theme(axis.line=element_blank(),
                axis.text.x=element_blank(),
                axis.text.y=element_blank(),
                axis.ticks=element_blank(),
                axis.title.x=element_blank(),
                axis.title.y=element_blank(),
                legend.position="none",
                #panel.background=element_blank(),
                #panel.border=element_blank(),
                panel.grid.minor=element_blank(),
                plot.background=element_blank(),
                plot.margin=unit(c(0,0,-0.5,0), "lines"))
      }
    }
   #ggsave("arrange.pdf", do.call(arrangeGrob, pltList))
   #figure1<-do.call(arrangeGrob, pltList)
   #ggsave(file="posterior.pdf",figure1)
   #n <- length(pltList)
   #nCol <- floor(sqrt(n))
   pdf('posterior.pdf')
   do.call("grid.arrange", pltList)
   dev.off()
   #g <- do.call("arrangeGrob", c(pltList, list(ncol=ncol(p_values_final)-1,main=textGrob("Model posterior", vjust=0.5, gp=gpar(fontsize=18, fontface="bold", fontsize=18)))))
   #ggsave(file="posterior.pdf", g, width=9, height=9, dpi=300)
}
#      }
#    }
#  pdf('posterior.pdf', width=9, height=9)
#  multiplot(plotlist = pltList, cols = numb_params+1)
#  dev.off()
#}

p_values_final = read.table("results_txt_files/Parameter_values_final.txt")
p_weights_final = read.table("results_txt_files/Parameter_weights_final.txt")
p_values_final <- subset(p_values_final, select = -p_values_final[,1] )
p_values_final$param_weights <- unlist(p_weights_final)

doc = xmlInternalTreeParse("input_file.xml")
top = xmlRoot(doc)
df <- xmlToDataFrame(top[["parameters"]])
lim <- df[-1, 3:4]
limits <- do.call(cbind, lapply(df[-1, 3:4], as.vector))
param_nam <- do.call(cbind, lapply(df[-1,1], as.character))
param_names <- c(param_nam,"weights")
colnames(p_values_final) = c(param_nam,"weights")
a=as.numeric(limits[,1])
b=as.numeric(limits[,2])

plot_posterior_distr(limits, param_names, p_values_final)
