#Rscript for exploring the effect of learnt language in the recall of specific words

#History_accuracy = history_correct/history_seen

library(dplyr)
library(ggplot2)
library(car)
library(pgirmess)
library(rstatix)

learning_traces.13m <- read.csv("C:/Users/migue/Documents/Miguel/Leuven/Datathon/learning_traces.13m.csv")

learning_traces.13m_ml <- learning_traces.13m %>%
  mutate(history_accuracy = history_correct / history_seen)

#Easiest language to learn given a certain language (group by ui language)
names(learning_traces.13m_ml)

#Subset for user interface language English
English <- subset(learning_traces.13m_ml, learning_traces.13m_ml$ui_language == 'en') 

E_by_learnt_language <- group_by(English, learning_language)
Avg_history_accuracy= mean(E_by_learnt_language$history_accuracy)

#E_by_learnt_language <- summarise(E_by_learnt_language, Avg_history_accuracy= mean(history_accuracy))
#A tibble: 5 × 2
# learning_language Avg_p_recall
# <chr>                            <dbl>
# 1 de                               0.903
# 2 es                               0.901
# 3 fr                               0.889
# 4 it                               0.900
# 5 pt                               0.907

#ANOVA attempt for how much the learnt language impacts word recall

E_by_learnt_language$learning_language <- factor(E_by_learnt_language$learning_language)

Eng.aov1 <- aov(history_accuracy ~ learning_language, data = E_by_learnt_language)
summary(Eng.aov1)
#Df Sum Sq Mean Sq F value Pr(>F)    
# learning_language       4    235   58.77    2966 <2e-16 ***
#   Residuals         7839430 155312    0.02                   
# ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1


#Check ANOVA assumptions

#Homocedasticity

leveneTest(E_by_learnt_language$history_accuracy ~ E_by_learnt_language$learning_language)
# Levene's Test for Homogeneity of Variance (center = median)
#            Df F value    Pr(>F)    
# group       4  1705.8 < 2.2e-16 ***
#       7839430              

#Unequal variances, use modified ANOVA 

#Normality
set.seed(123)

res <- residuals(Eng.aov1)

res_sample <- sample(res, 5000)

qqnorm(res_sample)
qqline(res_sample, col = "red")

#Data is nor normal, try the Non-parametric test Kruskal-Wallis


kruskal.test(history_accuracy ~ learning_language, data = E_by_learnt_language)
#Kruskal-Wallis rank sum test

# data:  p_recall by learning_language
# Kruskal-Wallis chi-squared = 10795, df = 4, p-value < 2.2e-16

#Check the pairwise comparisons
kruskalmc(E_by_learnt_language$history_accuracy, E_by_learnt_language$learning_language)
#Multiple comparison test after Kruskal-Wallis 
# alpha: 0.05 
# Comparisons
# obs.dif critical.dif stat.signif
# de-es  47528.623     6294.629        TRUE
# de-fr 323770.358     7022.601        TRUE
# de-it  52156.194     8866.109        TRUE
# de-pt  82933.523    12543.332        TRUE
# es-fr 276241.735     5777.412        TRUE
# es-it   4627.571     7916.404       FALSE
# es-pt 130462.146    11891.036        TRUE
# fr-it 271614.164     8506.704        TRUE
# fr-pt 406703.881    12291.920        TRUE
# it-pt 135089.717    13430.647        TRUE

#As everything is significant due to the big sample size, we will focus on the effect sizes to find differences

p1 <- kruskal_effsize(E_by_learnt_language,
                      E_by_learnt_language$history_accuracy ~ E_by_learnt_language$learning_language)
p1
# A tibble: 5 × 6
# learning_language .y.                                         n effsize method  magnitude
# * <fct>             <chr>                                   <int>   <dbl> <chr>   <ord>    
#   1 de                E_by_learnt_language$history_accuracy 1452597 0.0200  eta2[H] small    
# 2 es                E_by_learnt_language$history_accuracy 3407689 0.00852 eta2[H] small    
# 3 fr                E_by_learnt_language$history_accuracy 1873734 0.0155  eta2[H] small    
# 4 it                E_by_learnt_language$history_accuracy  793935 0.0366  eta2[H] small    
# 5 pt                E_by_learnt_language$history_accuracy  311480 0.0932  eta2[H] moderate

#English speakers recall Portuguese words slightly better than other languages.

#Violin plot
stats_labels <- data.frame(
  learning_language = c("de", "es", "fr", "it", "pt"),
  eff_size = c(0.0200, 0.0085, 0.0155, 0.0366, 0.0932),
  p_val = "< 0.001"
) %>%
  mutate(label = paste0("eta2: ", eff_size, "\np ", p_val))

set.seed(235)
sample_data <- E_by_learnt_language[sample(nrow(E_by_learnt_language), 5000), ]

p2 <- ggplot(sample_data, aes(x = learning_language, y = history_accuracy, fill = learning_language)) +
  geom_violin(trim = FALSE, alpha = 0.5) +
  stat_summary(fun = mean, geom = "point", color = "black", size = 2) +
  stat_summary(fun.data = mean_cl_boot, geom = "errorbar", width = 0.2) +
  geom_text(data = stats_labels, 
            aes(label = label, y = 1.1), 
            size = 3.5, 
            fontface = "bold",
            vjust = 0) +
  theme_minimal() +
  guides(fill = "none") +
  labs(title = "Memory Retention by Language for English Speaking Users",
       subtitle = "Kruskal-Wallis Effect Size (eta2) and p-values") + 
  ylab(label = "History accuracy") + 
  xlab(label = "Language learnt") +
  # Slightly increase label sizes
  theme(
    axis.title.x = element_text(size = 16), # Increases "Language learnt"
    axis.title.y = element_text(size = 16), # Increases "History accuracy"
    axis.text.x = element_text(size = 15),                # Increases language names
    axis.text.y = element_text(size = 15), plot.title = element_text(size = 16, face = "bold"),     # Main title size
    plot.subtitle = element_text(size = 12),                # Subtitle size                 # Increases accuracy numbers
  ) 
p2
