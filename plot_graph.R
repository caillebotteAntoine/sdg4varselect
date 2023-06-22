rm(list = ls())
graphics.off()

require(dplyr)
require(ggplot2)
require(reshape2)
#setwd("~/")


root <- "//wsl.localhost/Ubuntu/home/acaillebotte/projects/sdg4varselect/"

folder <- paste0(root,"joint_model_100")

#folder <- 'C:/Users/acaillebotte/Documents/plot_BEAMER_23_05/save_100'


dt <- read.csv2(paste0(folder, "/images/penalized_estimate_theta.csv"), dec = ".") %>%
  filter(variable %in% c('mu1','sigma2', 'gamma2_2','alpha'))


gg <- dt %>% ggplot(aes(value,variable, fill = variable)) +
  geom_violin() +
  geom_boxplot(width = 0.3, col = 'black', size = 0.5) +

  theme(axis.text.y = element_blank(), axis.ticks.y = element_blank()) +

  facet_wrap( vars(variable), scales = 'free') +
  theme(legend.position = 'null') +
  theme(axis.text.x = element_text(angle = 45, hjust = 1) ) +
  labs(x = '', y = '')

gg <- gg +
  geom_vline(data = data.frame(variable = unique(dt$variable),
                               value = c(0.3, 20,0.001,11.11)),#c(0.3,90.0,7.5, 0.0025, 20, 0.001, 110.11)),
             aes(xintercept = value))#shape = 8, col = variable) )

gg
ggsave(paste0(folder,'images/violin_penalized_plot.png'), gg, width = 8.2 , height = 5.6)

#==============================================================================#

dt <- read.csv2(paste0(folder, "/images/estimate_theta.csv"), dec = ".") %>%
  filter(variable %in% c('mu1','sigma2', 'gamma2_2','alpha'))

gg <- dt %>% ggplot(aes(value,variable, fill = variable)) +
  geom_violin() +
  geom_boxplot(width = 0.3, col = 'black', size = 0.5) +
  
  theme(axis.text.y = element_blank(), axis.ticks.y = element_blank()) +
  
  facet_wrap( vars(variable), scales = 'free') +
  theme(legend.position = 'null') +
  theme(axis.text.x = element_text(angle = 45, hjust = 1) ) +
  labs(x = '', y = '')

gg <- gg +
  geom_vline(data = data.frame(variable = unique(dt$variable),
                               value = c(0.3, 20,0.001,11.11)),#c(0.3,90.0,7.5, 0.0025, 20, 0.001, 110.11)),
             aes(xintercept = value))#shape = 8, col = variable) )

gg
ggsave(paste0(folder,'images/violin_unpenalized_plot.png'), gg, width = 8.2 , height = 5.6)
#==============================================================================#

dt_beta <- read.csv2(paste0(folder, "/images/penalized_estimate_beta.csv"), dec = ".")

n_run <- nrow(dt_beta)/length(unique(dt_beta$variable))
p <- length(unique(dt_beta$variable))
dt_beta$variable <- factor(rep(1:p, each = n_run))
dt_beta$id_run <- factor(rep(1:n_run, times = p))

dt_beta <- dt_beta[dt_beta$value != 0,]



gg <- dt_beta %>%

  ggplot(aes(variable, value)) + geom_boxplot(aes( fill = variable)) +
  theme(legend.position = 'null') +
  labs(x = 'Selected variables')

gg <- gg +
  geom_point(data = data.frame(variable = unique(dt_beta$variable),
                               value = c(-2,-1,1,2, rep(0, dt_beta$variable %>% unique %>% length - 4) )),
             shape = 8, aes(col = variable), size = 8 )+
  geom_point(data = data.frame(variable = unique(dt_beta$variable),
                               value = c(-2,-1,1,2, rep(0, dt_beta$variable %>% unique %>% length - 4) )),
             aes(col = variable), size = 6 )

gg <- gg + theme(axis.text.x = element_text(size = 16))
gg

ggsave(paste0(folder,'images/beta_penalized.png'), gg, width = 8.2 , height = 5.6)

#==============================================================================#



dt_beta <- read.csv2(paste0(folder, "/images/estimate_beta.csv"), dec = ".")

n_run <- nrow(dt_beta)/length(unique(dt_beta$variable))
p <- length(unique(dt_beta$variable))
dt_beta$variable <- factor(rep(1:p, each = n_run))
dt_beta$id_run <- factor(rep(1:n_run, times = p))

dt_beta <- dt_beta[dt_beta$value != 0,]



gg <- dt_beta %>%

  ggplot(aes(variable, value)) + geom_boxplot(aes( fill = variable)) +
  theme(legend.position = 'null') +
  labs(x = 'Selected variables')

gg <- gg +
  geom_point(data = data.frame(variable = unique(dt_beta$variable),
                               value = c(-2,-1,1,2, rep(0, dt_beta$variable %>% unique %>% length - 4) )),
             shape = 8, aes(col = variable), size = 8 )+
  geom_point(data = data.frame(variable = unique(dt_beta$variable),
                               value = c(-2,-1,1,2, rep(0, dt_beta$variable %>% unique %>% length - 4) )),
             aes(col = variable), size = 6 )

gg <- gg + theme(axis.text.x = element_text(size = 16))

gg
ggsave(paste0(folder,'images/beta_unpenalized.png'), gg, width = 8.2 , height = 5.6)


#==============================================================================#
# 
# mysize <- 7
# 
# prox <- function(beta,lbd){
#   if(abs(beta) < lbd) 
#     return(0)
#   
#   if(beta >= lbd) return(beta - lbd)
#   if(beta <= lbd) return(beta + lbd)
# }
# 
# lbd <- 0.5
# dt = data.frame(beta = seq(-2,2, length.out = 100 )) %>% mutate(prox  = beta %>% sapply(prox, lbd))
# 
# gg <- dt %>% ggplot(aes(beta, prox)) +
#   geom_line(col = "red", linewidth = 1) +
#   geom_line(col = "blue", aes(beta, beta), linewidth = 1) +
#   
#   xlim(x=c(-1,1)) + ylim(y=c(-1,1)) +
#   geom_segment(aes(x = -lbd,xend = -lbd, y = 0, yend = 0.1), linetype = 'dotted', linewidth = 1) +
#   annotate('text', x = -lbd-0.1/2, y = .15, label = expression(~-lambda), size = mysize) + 
#   
#   
#   geom_segment(aes(x = lbd,xend = lbd, y = 0, yend = 0.1), linetype = 'dotted', linewidth = 1) +
#   annotate('text', x = lbd-0.1/2, y = .15, label = expression(~+lambda), size = mysize) +
#   
#   
#   annotate('text', x = 0.75, y = 0.85, label = expression(beta), size = mysize, colour = 'blue')+
#   annotate('text', x = 0.75, y = 0.5, label = expression(prox(beta)), size = mysize, colour = 'red') +
#   
#   labs(x=expression(beta),y=expression(prox(beta)))
# 
# 
# gg
# ggsave(paste0(folder,'images/proximal_operator.png'), gg, width = 8.2 , height = 5.6)
  

# logistic <- function(t, phi1, phi2, phi3) return(phi1/(1+exp((phi2-t)/phi3)))
# 
# dt <- rbind(data.frame(x = seq(0, 10, by = 0.1), type = 1) %>%
#               mutate(logistic =  x %>% sapply(logistic, 0.8, 5, 0.6)),
#             
#             data.frame(x = seq(0, 10, by = 0.1), type = 2) %>%
#               mutate(logistic =  x %>% sapply(logistic, 0.5, 3, 0.4)))
# 
# gg <- dt %>% ggplot(aes(x, logistic)) + geom_line(aes( col = factor(type)), linewidth= 1) +
#   theme(legend.position = 'None') +
#   labs(y = '',x = '') + 
#   
#   geom_hline(yintercept = 0.8, linetype = 'dashed', linewidth = 1 )  +
#   annotate('text', x = 0.5, y = 0.75, label = expression(phi[1]), size = mysize) +
#   
#   geom_vline(xintercept = 5, linetype = 'dashed', linewidth = 1 )  +
#   annotate('text', x = 5.5, y = 0.1, label = expression(phi[2]), size = mysize)  
# 
# slope = 0.28
# gg <- gg +geom_segment(aes(x = 3.75, xend = 6.25, 
#                     y = logistic(5,0.8, 5, 0.6)-slope*1.25, yend = logistic(5,0.8, 5, 0.6)+slope*1.25),
#                 linetype = 'dashed', linewidth = 1) +
#   annotate('text', x = 6.8, y = 0.7, label = expression(phi[3]), size = mysize)  
# 
# gg
# ggsave(paste0(folder,'images/logistic.png'), gg, width = 8.2 , height = 5.6)
# 
# 
# 
# 
# n <- 100
# dt <- data.frame(phi1 <- rnorm(n, mean = ))
# 
# library(stringr)
# 
# dt <- read.csv2("prev_2021.csv"), sep = ";", dec = ".")
# 
# data <- dt %>% melt(id = c("module","genotype","Y","X" )) %>%mutate(date = as.Date(variable, format = 'W_%d.%m.%y')) 
# #%>%  mutate(module = factor(module))
# 
# gg <- data[!grepl("_TT", data$variable),] %>% filter(module == 'date 1') %>% filter(X < 5) %>%
#   ggplot(aes(date, value, col = interaction(genotype, Y, X))) +
#   geom_line(linewidth = 1) +
#   theme(legend.position = 'None') +
#   labs(x = 'Date', y = 'Prevalence')
# 
# 
# 
# ggsave(paste0(folder,'images/prevalence.png'), gg, width = 8.2 , height = 5.6)






# 
# 
# 
# 
# 
# 
# 
# mysize <- 7
# logistic <- function(t, phi1, phi2, phi3) return(phi1/(1+exp((phi2-t)/phi3)))
# 
# dt <- rbind(data.frame(x = seq(0, 5, by = 0.1), type = 1) %>%
#               mutate(logistic =  x %>% sapply(logistic, 0.8, 5, 0.6)),
#             data.frame(x = seq(5, 10, by = 0.1), type = 2) %>%
#               mutate(logistic =  x %>% sapply(logistic, 0.8, 5, 0.6)) )
# 
# 
# 
# gg <- dt %>% ggplot(aes(x, logistic)) + geom_line(aes( col = factor(type)), linewidth= 1) +
#   theme(legend.position = 'None') +
#   labs(y = '',x = '') +
#   xlim(0,10) + ylim(0,1) +
# 
#   geom_segment(aes(x = 1,xend = 3, y = 0.25, yend = logistic(3)), 
#                linewidth = 1, arrow = arrow(length = unit(0.03, "npc")) +
#                  
#   annotate('text', x = 1, y = .28, label = "known information", size = mysize) +
# 
# gg
# 
# 
# gg
# ggsave(paste0(folder,'images/logistic.png'), gg, width = 8.2 , height = 5.6)




