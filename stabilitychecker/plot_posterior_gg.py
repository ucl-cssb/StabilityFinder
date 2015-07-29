import ggplot


def plot_posterior_distr(limits, param_names):

  #p_values_final = read.table(paste("results_txt_files/Parameter_values_final.txt",sep=''))
  #p_weights_final = read.table(paste("results_txt_files/Parameter_weights_final.txt",sep=''))
  #p_values_final <- subset(p_values_final, select = -p_values_final[,1] )
  #p_values_final$param_weights <- unlist(p_weights_final)
  numb_params = len(param_names)-1

  a=limits[1,]
  b=limits[2,]
  pltList =[]
  #library(gridExtra)
  k=0
  for i in range(1,numb_params):
    for j in range(1,numb_params):
      k += 1
      print k
      if i==j:
        pltList[k].append(ggplot(p_values_final, aes_string(x=param_names[i], weight=param_names[ncol(p_values_final)])) + geom_density(fill="grey") + xlim(a[i],b[i])+ ggtitle(param_names[i]) +
          theme(axis.line=element_blank(),
                plot.title=element_text(size=8, hjust=0,lineheight=0),
                axis.text.x=element_text(size=6,angle = 90, vjust=0,hjust=1.2),
                axis.text.y=element_text(size=6),
                axis.ticks=element_blank(),
                axis.title.x=element_blank(),
                axis.title.y=element_blank(),
                legend.position="none",
                panel.grid.minor=element_blank(),
                plot.background=element_blank()))
                #plot.margin=unit(c(0,0,-0.5,0), "lines"))
      else:
        pltList[k].append(ggplot(p_values_final, aes_string(x = param_names[i], y = param_names[j], weight=param_names[ncol(p_values_final)])) + xlim(a[i],b[i])+ ylim(a[j],b[j])+
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
                plot.background=element_blank()))
                #plot.margin=unit(c(0,0,-0.5,0), "lines"))


  #g <- do.call("arrangeGrob", c(pltList, list(ncol=ncol(p_values_final)-1, main=textGrob("bla", vjust=0.5, gp=gpar(fontsize=18, fontface="bold", fontsize=18)))))
  #ggsave(file=paste(results_path,"/",pop_fold_res_path,"/posterior.pdf",sep=''), g,width=9, height=9, dpi=300)


p_values_final = read.table(paste("results_txt_files/Parameter_values_final.txt",sep=''))
p_weights_final = read.table(paste("results_txt_files/Parameter_weights_final.txt",sep=''))
p_values_final <- subset(p_values_final, select = -p_values_final[,1] )
p_values_final$param_weights <- unlist(p_weights_final)
colnames(p_values_final) <- c("a1", "beta","a2","gama","weights")
param_names <- c("a1", "beta","a2","gama","weights")
limits <- cbind(c(0,10),c(0,10),c(0,10),c(0,10) )
plot_posterior_distr(limits, param_names)
