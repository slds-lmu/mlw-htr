library(readr)
library(readxl)
#synthetic_train <- read_csv("C:/Users/ru84fuj/Desktop/results/3/synthetic_train.csv")
#benchmark <- read_excel("C:/Users/ru84fuj/Desktop/results/benchmark_own_vision_tesseract_40000.xlsx")
benchmark1 <- read_excel("/home/USER/Documents/Uni/WiSe2223/Consulting/mlw-consulting-project/data/processed/results_swin_gpt2.xlsx")
# benchmark2 <- read_excel("/home/USER/Documents/Uni/WiSe2223/Consulting/mlw-consulting-project/data/processed/cloud_latin_bench.xlsx")
benchmark3 <- read_excel("/home/USER/Documents/Uni/WiSe2223/Consulting/mlw-consulting-project/data/processed/predictions_tess.xlsx")
benchmark4 <- read_excel("/home/USER/Documents/Uni/WiSe2223/Consulting/mlw-consulting-project/data/processed/test_predictions_swin_gpt2_aug_abl.xlsx")
benchmark5 <- read_excel("/home/USER/Documents/Uni/WiSe2223/Consulting/mlw-consulting-project/data/processed/predictions_easyocr.xlsx")
benchmark6 <- read_excel("/home/USER/Documents/Uni/WiSe2223/Consulting/mlw-consulting-project/data/processed/predictions_paddleocr.xlsx")
benchmark <- rbind(benchmark1, benchmark3, benchmark4, benchmark5, benchmark6)

benchmark$Model <- factor(benchmark$Model, levels=unique(benchmark$Model))
benchmark$CER <- round(benchmark$CER, 5)
benchmark$Correct <- ifelse(benchmark$CER == 0, 1, 0)
# Exclude GPT2
#benchmark <-benchmark[benchmark$Model != 'Own (ViT+GPT2)' & benchmark$Model != 'Own (ViT+pretrained GPT2)',]

library(ggplot2)
library(dplyr)
library(scales)
library(grid)
library(gridExtra)
library(ggthemes)

summary_df <- benchmark %>% group_by(Model) %>% summarize(Examples = n(),
                                                           Mean = round(mean(CER),4), 
                                                           Median = round(median(CER),4),
                                                           Min = round(min(CER), 4),
                                                           Max = round(max(CER),4),
                                                           StdDev = round(sd(CER),4),
                                                           'Correctly predicted labels (%)' = round(mean(Correct),4)*100) %>%
  as.data.frame()


#colnames(summary_df)<-c("Dataset", "Number of examples", "Mean", "Median", "Min", "Max", "Standard Deviation")

cer_plot<-ggplot(benchmark, aes(x = Model, y = CER, fill = Model))+geom_boxplot()+
  xlab("Model")+ylab("CER")+ylim(c(0,1))+labs(title = "Model benchmarking", 
                                   subtitle = paste0("CER over ", nrow(benchmark)/length(unique(benchmark$Model)), " test examples*"),
                                   #caption = "*Real data include inaccurate labels (10%) - relabeling ongoing"
  )+theme_economist() + scale_fill_economist()


density_plot<-ggplot(benchmark, aes(x = CER, fill = Model))+geom_density(alpha = 0.50)+
  xlab("CER")+ylab("Frequency")+xlim(c(0,1))+labs(title = "Model performance", 
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

