# roc_comparison.R
# Compares HLR and BBB ROC curves. Each model is evaluated against its own
# ground truth (round(p)) independently — the files don't need the same rows.

library(pROC)
library(data.table)   # install.packages("data.table") if needed
library(ggplot2)

HLR_FILE <- r"(results\hlr.settles.acl16.learning_traces.13m.preds)"
BBB_FILE <- r"(predictions_bbb.tsv)"

# fread with quote="" prevents apostrophes in lexeme strings from truncating the read
hlr <- fread(HLR_FILE, sep = '\t', na.strings = "NA", quote = "")
bbb <- fread(BBB_FILE, sep = '\t', na.strings = "NA", quote = "")

cat(sprintf("HLR rows: %d\n", nrow(hlr)))
cat(sprintf("BBB rows: %d\n", nrow(bbb)))

# Drop rows with NA in p or pp before computing ROC
hlr <- hlr[!is.na(hlr$p) & !is.na(hlr$pp)]
bbb <- bbb[!is.na(bbb$p) & !is.na(bbb$pp)]

cat(sprintf("HLR rows after NA removal: %d\n", nrow(hlr)))
cat(sprintf("BBB rows after NA removal: %d\n", nrow(bbb)))

roc_hlr <- roc(round(hlr$p) ~ hlr$pp, quiet = TRUE)
roc_bbb <- roc(round(bbb$p) ~ bbb$pp, quiet = TRUE)

auc_hlr <- as.numeric(auc(roc_hlr))
auc_bbb <- as.numeric(auc(roc_bbb))

cat(sprintf("HLR AUC: %.4f\n", auc_hlr))
cat(sprintf("BBB AUC: %.4f\n", auc_bbb))

ggroc(list("BBB" = roc_bbb, "HLR" = roc_hlr), linewidth = 3, legacy.axes = TRUE)+
  geom_abline(intercept = 0, slope = 1, linewidth = 1, color = "black", )+
  scale_colour_manual(
    values = c(
      "BBB" = "#E74C3C",
      "HLR" = "#2980B9"),
    labels = c(
      sprintf("BBB  (AUC = %.2f)", auc_bbb),
      sprintf("HLR  (AUC = %.2f)", auc_hlr)
    )
  ) +
  labs(title = "ROC Curve: HLR vs Bayesian Beta-Binomial", 
        x = "False Positive Rate (1 - Specificity)", 
        y = "True Positive Rate (Sensitivity)")+
  theme(
    axis.title.x = element_text(size = 16),
    axis.title.y = element_text(size = 16),
    axis.text.x = element_text(size = 15),
    axis.text.y = element_text(size = 15), 
    plot.title = element_text(size = 16, face = "bold"),
    plot.subtitle = element_text(size = 12))+
  theme_minimal()  


