library(dplyr)
library(readr)
library(readxl)
#synthetic_train <- read_csv("C:/Users/ru84fuj/Desktop/results/3/synthetic_train.csv")
#benchmark <- read_excel("C:/Users/ru84fuj/Desktop/results/benchmark_own_vision_tesseract_40000.xlsx")
benchmark <- read_excel("/home/USER/Documents/Uni/WiSe2223/Consulting/mlw-consulting-project/data/processed/all_modelen.xlsx")


# Custom Modifications
new_ours <- "Ours (Best)"
new_gcv <- "Google Cloud Vision"
benchmark$Model[benchmark$Model == "Google Cloud Vision + Post-processing"] <- new_gcv
benchmark$Model[benchmark$Model == "SWIN+GPT-2 (Best)"] <- new_ours

benchmark <- benchmark %>% filter(Model == new_ours | Model == new_gcv)

benchmark$Model <- factor(benchmark$Model, levels=unique(benchmark$Model))
benchmark$CER <- round(benchmark$CER, 5)
# Exclude GPT2
#benchmark <-benchmark[benchmark$Model != 'Own (ViT+GPT2)' & benchmark$Model != 'Own (ViT+pretrained GPT2)',]

library(ggplot2)
library(scales)
library(grid)
library(gridExtra)
library(ggthemes)

summary_df <- benchmark %>% group_by(Model) %>% summarize(Examples = n(),
                                                           Mean = round(mean(CER),4),
                                                           WeightedCER = round(sum(`Weighted CER`)/sum(nchar(Label)), 4),
                                                           Median = round(median(CER),4),
                                                           Min = round(min(CER), 4),
                                                           Max = round(max(CER),4),
                                                           StdDev = round(sd(CER),4),
                                                           'Correctly predicted labels (%)' = round(mean(Correct),4)*100) %>%
  as.data.frame()


#colnames(summary_df)<-c("Dataset", "Number of examples", "Mean", "Median", "Min", "Max", "Standard Deviation")

cer_plot<-ggplot(benchmark, aes(x = Model, y = CER, fill = Model))+geom_violin()+
  xlab("Model")+ylab("CER")+labs(title = "Model benchmarking", 
                                   subtitle = paste0("CER over ", nrow(benchmark)/length(unique(benchmark$Model)), " test examples*"),
                                   #caption = "*Real data include inaccurate labels (10%) - relabeling ongoing"
  )+theme_economist() + scale_fill_economist()


density_plot<-ggplot(benchmark, aes(x = CER, fill = Model))+geom_density(alpha = 0.50)+
  xlab("CER")+ylab("Frequency")+xlim(c(0,1))+labs(title = "Model benchmarking",
                                                  subtitle = paste0("CER over ", nrow(benchmark)/length(unique(benchmark$Model)), " test examples*"))+theme_economist() + scale_fill_economist()



caption = "*All predictions were performed over real images"



tt <- ttheme_default(colhead=list(fg_params = list(parse=TRUE)),
                     base_size = 12,
                     padding = unit(c(20, 4), "mm"))

tbl <- tableGrob(summary_df, rows=NULL, theme=tt)

grid.arrange(cer_plot, tbl, 
             nrow = 2, heights = c(4, 2))

grid.arrange(density_plot, tbl, 
             nrow = 2, heights = c(4, 2))

