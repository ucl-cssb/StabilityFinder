library(ggplot2)
library(gridExtra)

#---------------------------------------------------------------------------------------------------
plot_stabilityChecker_particles <- function(numb_files, filename, title_text){
filelist <- list.files(pattern = "set_result*")
data_list = lapply(filelist, read.table, sep = " ")
pltList <- list()
for(i in 1:numb_files){
  pltList[[i]] <-ggplot(data_list[[i]], aes(x=V1, y=V2)) + geom_point() + xlim(0,25) + ylim(0,25) + theme(axis.title.x = element_blank(),
                                                                                                           axis.title.y = element_blank())

}
g <- do.call("arrangeGrob", c(pltList, list(ncol=sqrt(numb_files) ,sub=textGrob("a",gp = gpar(fontsize=50, fontface="bold", fontsize=50)), left=textGrob("b", rot = 90, vjust = 1,gp = gpar(fontsize=20, fontface="bold", fontsize=20)),main=textGrob(title_text, vjust =1, gp = gpar(fontsize=20, fontface="bold", fontsize=20)))))
ggsave(file=paste(filename,".pdf",sep=''), g,width=12, height=12, dpi=300)
}
plot_stabilityChecker_particles(10, "switch_result","Steady state values of the particles in the last population of the bistable toggle switch")

#---------------------------------------------------------------------------------------------------
p_values_final = read.table("Parameter_values_final.txt")
p_weights_final = read.table("Parameter_weights_final.txt")
p_values_final <- subset(p_values_final, select = -p_values_final[,1] )
p_values_final$param_weights <- unlist(p_weights_final)
colnames(p_values_final) <- c("ge", "rep","rep_r","dim","dim_r","deg","rep_dim","rep_dim_r","deg_sr", "deg_dim","weights")
p1 <- ggplot(p_values_final, aes(x=ge, weight=weights)) + geom_density(fill="grey")+ xlim(1,10) 
p2 <- ggplot(p_values_final, aes(rep, weight=weights))  + geom_density(fill="grey") + xlim(1,10) 
p3 <- ggplot(p_values_final, aes(rep_r, weight=weights))  + geom_density(fill="grey") + xlim(1,10) 
p4 <- ggplot(p_values_final, aes(dim, weight=weights))  + geom_density(fill="grey") + xlim(1,10) 
p5 <- ggplot(p_values_final, aes(dim_r, weight=weights) ) + geom_density(fill="grey") + xlim(0,5) 
p6 <- ggplot(p_values_final, aes(deg, weight=weights) ) + geom_density(fill="grey") + xlim(1,10) 
p7 <- ggplot(p_values_final, aes(rep_dim, weight=weights))  + geom_density(fill="grey") + xlim(5,15) 
p8 <- ggplot(p_values_final, aes(rep_dim_r, weight=weights) ) + geom_density(fill="grey") + xlim(0.001,0.1) 
p9 <- ggplot(p_values_final, aes(deg_sr, weight=weights) ) + geom_density(fill="grey") + xlim(0.01,0.1) 
p10 <- ggplot(p_values_final, aes(deg_dim, weight=weights) ) + geom_density(fill="grey") + xlim(0,0.5) 
grid.arrange(p1, p2, p3, p4, p5,p6,p7,p8,p9,p10,ncol=2) #arranges plots within grid
g <- arrangeGrob(p1, p2, p3, p4, p5,p6,p7,p8,p9,p10, ncol=2,main=textGrob('Parameter value densities of last population of the toggle switch', vjust =1, gp = gpar(fontsize=14, fontface="bold", fontsize=14))) #generates g
ggsave(file="param_densities_last_pop.pdf", g, scale=2)
#---------------------------------------------------------------------------------------------------
plot_posterior_distr <- function(numb_params, param_names, limits, file_name,plot_title){
  a=limits[1,]
  b=limits[2,]
  pltList <- list()
  library(gridExtra)
  k=0
  for(i in 1:numb_params)
    for(j in 1:numb_params){
      k=k+1
      print(k)
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
      }
      else{
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
                panel.background=element_blank(),
                panel.border=element_blank(),
                panel.grid.minor=element_blank(),
                plot.background=element_blank(),
                plot.margin=unit(c(0,0,-0.5,0), "lines"))
      }
    } 
  g <- do.call("arrangeGrob", c(pltList, list(ncol=ncol(p_values_final)-1,main=textGrob(plot_title, vjust=0.5, gp=gpar(fontsize=18, fontface="bold", fontsize=18)))))
  ggsave(file=paste(file_name,".png",sep=''), g,width=9, height=9, dpi=300)
}

#param_names <- c("ge", "rep","rep_r","dim","dim_r","deg","rep_dim","rep_dim_r","deg_sr", "deg_dim","weights")
#limits <- cbind(c(1,10),c(1,10),c(1,10),c(1,10),c(0,5),c(1,10),c(5,15),c(0.001,0.1),c(0.01,0.1),c(0,0.5) )
#file_name <- "Toggle_switch_posterior"
#plot_title <- "Toggle switch posterior distribution"
p_values_final = read.table("Parameter_values_final.txt")
p_weights_final = read.table("Parameter_weights_final.txt")
p_values_final <- subset(p_values_final, select = -p_values_final[,1] )
p_values_final$param_weights <- unlist(p_weights_final)
colnames(p_values_final) <- c("a1", "beta","a2","gama","weights")

param_names <- c("a1", "beta","a2","gama","weights")
limits <- cbind(c(0,10),c(0,10),c(0,10),c(0,10) )
plot_posterior_distr(4, param_names, limits, "Gardner_posterior", "Gardner switch posterior distribution")




