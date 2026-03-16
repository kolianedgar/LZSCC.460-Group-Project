# ==============================================================================
# 04a_xgboost_sample.R
# XGBoost Predictive Model for Customer Conversion
# LZSCC.460 Group Project - Objective 4
#
# Imbalance strategy: observation weights + sample-based scale_pos_weight
#   scale_pos_weight = n_neg / n_pos (derived from training set class counts)
# ==============================================================================


## Library Implementation Stage

library(xgboost)
library(dplyr)
library(caret)
library(pROC)
library(ggplot2)
library(SHAPforxgboost)


## Setup Stage

source("utils/utils_preprocessing.R")
source("utils/utils.R")
source("01_preprocessing.R")

if (!dir.exists("plots - Enes/xgboost")) dir.create("plots - Enes/xgboost", recursive = TRUE)


## Importing & Preprocessing Stage
# Uses preprocess_data() which handles: duplicate removal, ID removal,
# ConversionRate filtering, stratified split, imputation, outlier capping,
# encoding, matrix conversion, and class weight computation.

raw_data <- read.csv("marketing.csv")

processed <- preprocess_data(
  data        = raw_data,
  target_col  = "Conversion",
  id_cols     = "CustomerID",
  filter_col  = "ConversionRate",
  filter_min  = 0,
  filter_max  = 1,
  train_prop  = 0.8,
  stratify    = TRUE,
  seed        = 123,
  model_type  = "xgb"
)

X_train     <- processed$X_train
y_train     <- as.numeric(as.character(processed$y_train))
X_test      <- processed$X_test
y_test      <- as.numeric(as.character(processed$y_test))
obs_weights <- processed$obs_weights

cat("Train size:", dim(X_train), "\n")
cat("Test size: ", dim(X_test), "\n")


## Class Imbalance Setup

n_pos <- sum(y_train == 1)
n_neg <- sum(y_train == 0)
spw   <- n_neg / n_pos

cat("\nClass 0 (non-converters):", n_neg, sprintf("(%.1f%%)", 100 * n_neg / length(y_train)))
cat("\nClass 1 (converters):    ", n_pos, sprintf("(%.1f%%)", 100 * n_pos / length(y_train)))
cat("\nscale_pos_weight:        ", round(spw, 4), "\n")

# DMatrix with observation weights
dtrain <- xgb.DMatrix(data = X_train, label = y_train, weight = obs_weights)
dtest  <- xgb.DMatrix(data = X_test,  label = y_test)


## Hyperparameter Tuning Stage (Random Search)
# All hyperparameters are determined here via random search with 5-fold CV.
# Following Bergstra & Bengio (2012): random search is more efficient
# than grid search in high-dimensional hyperparameter spaces.

set.seed(42)
n_iter <- 50

search_space <- data.frame(
  eta              = runif(n_iter, 0.01, 0.3),
  max_depth        = sample(3:10, n_iter, replace = TRUE),
  min_child_weight = sample(1:10, n_iter, replace = TRUE),
  subsample        = runif(n_iter, 0.5, 1.0),
  colsample_bytree = runif(n_iter, 0.5, 1.0),
  gamma            = runif(n_iter, 0, 5)
)

# Helper: runs random search for a given model configuration
run_tuning <- function(dtrain, search_space, spw_value = NULL, label = "Model") {
  
  n <- nrow(search_space)
  results <- data.frame(
    iter = 1:n, eta = numeric(n), max_depth = integer(n),
    min_child_weight = integer(n), subsample = numeric(n),
    colsample_bytree = numeric(n), gamma = numeric(n),
    best_nrounds = integer(n), cv_logloss = numeric(n)
  )
  
  cat(sprintf("\nTuning %s (%d iterations)...\n", label, n))
  
  for (i in seq_len(n)) {
    params_i <- list(
      objective = "binary:logistic", eval_metric = "logloss",
      eta = search_space$eta[i], max_depth = search_space$max_depth[i],
      min_child_weight = search_space$min_child_weight[i],
      subsample = search_space$subsample[i],
      colsample_bytree = search_space$colsample_bytree[i],
      gamma = search_space$gamma[i],
      nthread = parallel::detectCores() - 1
    )
    if (!is.null(spw_value)) params_i$scale_pos_weight <- spw_value
    
    set.seed(123)
    cv <- xgb.cv(params = params_i, data = dtrain, nrounds = 1000,
                 nfold = 5, stratified = TRUE, early_stopping_rounds = 30,
                 verbose = 0)
    
    iter_best <- cv$early_stop$best_iteration
    log_col   <- grep("test.*logloss_mean", names(cv$evaluation_log), value = TRUE)[1]
    
    results[i, ] <- list(i, search_space$eta[i], search_space$max_depth[i],
                         search_space$min_child_weight[i], search_space$subsample[i],
                         search_space$colsample_bytree[i], search_space$gamma[i],
                         iter_best, cv$evaluation_log[[log_col]][iter_best])
    
    if (i %% 10 == 0) cat(sprintf("  [%d/%d] Best so far: %.4f\n", i, n, min(results$cv_logloss[1:i])))
  }
  
  best <- results[which.min(results$cv_logloss), ]
  cat(sprintf("  Best: eta=%.3f, depth=%d, nrounds=%d, logloss=%.4f\n",
              best$eta, best$max_depth, best$best_nrounds, best$cv_logloss))
  list(results = results, best = best)
}

tune <- run_tuning(dtrain, search_space, spw_value = spw, label = "XGBoost (sample spw)")

write.csv(tune$results, "plots - Enes/xgboost/tuning_results_sample.csv", row.names = FALSE)


## Training Final Model Stage

params <- list(
  objective = "binary:logistic", eval_metric = "logloss",
  eta = tune$best$eta, max_depth = tune$best$max_depth,
  min_child_weight = tune$best$min_child_weight,
  subsample = tune$best$subsample,
  colsample_bytree = tune$best$colsample_bytree,
  gamma = tune$best$gamma,
  scale_pos_weight = spw,
  nthread = parallel::detectCores() - 1
)

model <- xgb.train(params = params, data = dtrain,
                   nrounds = tune$best$best_nrounds,
                   evals = list(train = dtrain, test = dtest), verbose = 0)


## Evaluation Stage

evaluate <- function(model, dtest, y_actual, label) {
  probs <- predict(model, dtest)
  preds <- ifelse(probs > 0.5, 1, 0)
  
  cm  <- confusionMatrix(factor(preds, levels = c("0","1")),
                         factor(y_actual, levels = c("0","1")), positive = "1")
  roc <- roc(y_actual, probs, quiet = TRUE)
  
  cat(sprintf("\n%s\n", label))
  cat(sprintf("  AUC-ROC:     %.4f\n", auc(roc)))
  cat(sprintf("  Accuracy:    %.4f\n", cm$overall["Accuracy"]))
  cat(sprintf("  Sensitivity: %.4f\n", cm$byClass["Sensitivity"]))
  cat(sprintf("  Specificity: %.4f\n", cm$byClass["Specificity"]))
  cat(sprintf("  F1:          %.4f\n", cm$byClass["F1"]))
  cat("\nConfusion Matrix:\n")
  print(cm$table)
  
  list(probs = probs, cm = cm, roc = roc)
}

cat("\n=== TEST PERFORMANCE ===\n")
eval_result <- evaluate(model, dtest, y_test, "XGBoost (obs_weights + sample spw)")

# Save metrics
metrics <- data.frame(
  Model       = "XGBoost (sample spw)",
  AUC_ROC     = round(auc(eval_result$roc), 4),
  Accuracy    = round(eval_result$cm$overall["Accuracy"], 4),
  Sensitivity = round(eval_result$cm$byClass["Sensitivity"], 4),
  Specificity = round(eval_result$cm$byClass["Specificity"], 4),
  F1          = round(eval_result$cm$byClass["F1"], 4)
)
cat("\n=== METRICS ===\n")
print(metrics, row.names = FALSE)
write.csv(metrics, "plots - Enes/xgboost/metrics_sample.csv", row.names = FALSE)


## Feature Importance Stage

imp <- xgb.importance(feature_names = colnames(X_train), model = model)

cat("\nTop 10 by Gain:\n"); print(head(imp, 10))

# Plot top 15
imp_top <- imp[1:min(15, nrow(imp)), ]
imp_top$Feature <- factor(imp_top$Feature, levels = rev(imp_top$Feature))

p_imp <- ggplot(imp_top, aes(x = Feature, y = Gain)) +
  geom_col(fill = "#2563eb") + coord_flip() +
  labs(title = "Feature Importance - XGBoost (sample spw)", x = NULL, y = "Gain") +
  theme_minimal()
ggsave("plots - Enes/xgboost/importance_sample.png", p_imp, width = 10, height = 6, dpi = 150)


## SHAP Interpretability Stage
# SHAP values explain how each feature contributes to individual predictions.
# Following Lundberg & Lee (2017).

shap <- shap.values(xgb_model = model, X_train = X_train)
shap_long <- shap.prep(shap_contrib = shap$shap_score,
                       X_train = as.data.frame(X_train))

p_shap <- shap.plot.summary(shap_long) +
  ggtitle("SHAP Summary - XGBoost (sample spw)") + theme_minimal()
ggsave("plots - Enes/xgboost/shap_sample.png", p_shap, width = 10, height = 7, dpi = 150)

cat("\nTop 10 by mean |SHAP|:\n")
print(head(sort(shap$mean_shap_score, decreasing = TRUE), 10))


## Visualisation Stage

# ROC Curve
png("plots - Enes/xgboost/roc_sample.png", width = 800, height = 600)
plot(eval_result$roc, col = "#2563eb", lwd = 2,
     main = sprintf("ROC Curve - XGBoost (sample spw, AUC=%.3f)", auc(eval_result$roc)))
abline(a = 0, b = 1, lty = 2, col = "grey50")
dev.off()

# Predicted probability distribution
p_probs <- ggplot(
  data.frame(prob = eval_result$probs, actual = factor(y_test)),
  aes(x = prob, fill = actual)
) +
  geom_histogram(bins = 50, alpha = 0.6, position = "identity") +
  scale_fill_manual(values = c("0" = "#ef4444", "1" = "#2563eb"),
                    labels = c("0" = "Non-converter", "1" = "Converter")) +
  labs(title = "Predicted Probability Distribution - XGBoost (sample spw)",
       x = "P(Conversion = 1)", y = "Count", fill = "Actual") +
  theme_minimal()
ggsave("plots - Enes/xgboost/probability_distribution_sample.png", p_probs, width = 9, height = 5, dpi = 150)


## Save Stage

xgb.save(model, "plots - Enes/xgboost/model_sample.json")

cat("\n=== COMPLETE ===\n")
cat(sprintf("scale_pos_weight used: %.4f (n_neg / n_pos)\n", spw))
cat("Outputs saved to plots - Enes/xgboost/\n")
