#-----------------------------------
# Load libraries and utilities
#-----------------------------------
library(dplyr)
library(ggplot2)
library(glmnet)
library(caret)
library(pROC)
library(PRROC)

source("utils/utils.R")
source("01_preprocessing.R")

#--------------------------------
# Change the working directory
#   to where the file is located
#--------------------------------
set_script_wd()

#-----------------------------------
# Load data
#-----------------------------------
marketing_data <- read.csv("marketing.csv")

#-----------------------------------
# Preprocessing
#-----------------------------------
prep_glmnet <- preprocess_data(
  data       = marketing_data,
  target_col = "Conversion",
  id_cols    = "CustomerID",
  filter_col = "ConversionRate",
  filter_min = 0,
  filter_max = 1,
  train_prop = 0.8,
  stratify   = TRUE,
  seed       = 123,
  model_type = "glmnet"
)

X_train     <- prep_glmnet$X_train
X_test      <- prep_glmnet$X_test
y_train     <- prep_glmnet$y_train
y_test      <- prep_glmnet$y_test
obs_weights <- prep_glmnet$obs_weights

#---------------------------------
# Elastic-net CV (alpha + lambda)
#---------------------------------
alpha_grid <- seq(0, 1, by = 0.1)

set.seed(123)
cv_models <- lapply(alpha_grid, function(a) {
  cv.glmnet(
    x           = X_train,
    y           = y_train,
    family      = "binomial",
    alpha       = a,
    weights     = obs_weights,
    standardize = FALSE,
    keep        = TRUE
  )
})

cv_errors   <- sapply(cv_models, function(m) min(m$cvm))
best_idx    <- which.min(cv_errors)
best_alpha  <- alpha_grid[best_idx]
best_lambda <- cv_models[[best_idx]]$lambda.min
best_cv     <- cv_models[[best_idx]]

lambda_idx <- which(best_cv$lambda == best_lambda)
cv_probs <- as.numeric(best_cv$fit.preval[, lambda_idx])
cv_probs <- 1 / (1 + exp(-cv_probs))  # sigmoid transformation

#-------------------
# Threshold Tuning
#-------------------
tune_threshold <- function(probs, y, grid = seq(0.05, 0.95, by = 0.01)) {
  scores <- sapply(grid, function(t) {
    preds <- factor(ifelse(probs > t, "1", "0"), levels = levels(y))
    confusionMatrix(preds, y, positive = "1")$byClass["Balanced Accuracy"]
  })
  grid[which.max(scores)]
}

best_threshold <- tune_threshold(cv_probs, y_train)

#--------------
# Final model
#--------------
final_model <- glmnet(
  x           = X_train,
  y           = y_train,
  family      = "binomial",
  alpha       = best_alpha,
  lambda      = best_lambda,
  weights     = obs_weights,
  standardize = FALSE
)

coef_opt <- coef(cv_models, s=)
rownames(final_model)
#------------------------------
# Test evaluation (ONLY here)
#------------------------------
prob_test <- as.numeric(predict(final_model, X_test, type = "response"))
pred_test <- factor(
  ifelse(prob_test > best_threshold, "1", "0"),
  levels = levels(y_train)
)

# Test weights (mirrors training imbalance correction)
test_class_weights <- compute_class_weights(y_test)
test_obs_weights   <- test_class_weights[as.character(y_test)]
test_obs_weights   <- test_obs_weights / sum(test_obs_weights)

#----------------------------------
# Imbalance-robust metrics
#----------------------------------

# 1. Weighted accuracy
weighted_accuracy <- sum(test_obs_weights * (pred_test == y_test))

# 2. Confusion matrix
cm <- confusionMatrix(pred_test, y_test, positive = "1")

# 3. AUC-ROC
roc_obj <- roc(y_test, prob_test, quiet = TRUE)
auc_roc  <- as.numeric(auc(roc_obj))

# 4. AUC-PR
pr_obj <- pr.curve(
  scores.class0 = prob_test[y_test == "1"],
  scores.class1 = prob_test[y_test == "0"],
  curve         = TRUE
)
auc_pr <- pr_obj$auc.integral

#----------------------------------
# Report
#----------------------------------
cat("=== Hyperparameters ===\n")
cat("Best Alpha:              ", round(best_alpha, 4),     "\n")
cat("Best Lambda:             ", round(best_lambda, 4),    "\n")
cat("Best Threshold (tau):    ", round(best_threshold, 4), "\n")

cat("\n=== Imbalance-Robust Evaluation ===\n")
cat("Weighted Accuracy:       ", round(weighted_accuracy,                      4), "\n")
cat("Balanced Accuracy:       ", round(cm$byClass["Balanced Accuracy"],        4), "\n")
cat("Sensitivity (Recall):    ", round(cm$byClass["Sensitivity"],              4), "\n")
cat("Specificity:             ", round(cm$byClass["Specificity"],              4), "\n")
cat("Precision (PPV):         ", round(cm$byClass["Pos Pred Value"],           4), "\n")
cat("F1 Score:                ", round(cm$byClass["F1"],                       4), "\n")
cat("AUC-ROC:                 ", round(auc_roc,                                4), "\n")
cat("AUC-PR:                  ", round(auc_pr,                                 4), "\n")

# Extract coefficients at best lambda
coef_mat <- coef(final_model, s = best_lambda)
coef_df <- data.frame(
  Feature    = rownames(coef_mat),
  Coefficient = as.numeric(coef_mat)
) |>
  filter(Feature != "(Intercept)", Coefficient != 0) |>  # drop intercept & zero-coef (elastic-net zeroed)
  mutate(
    Direction = ifelse(Coefficient > 0, "Positive", "Negative"),
    Feature   = reorder(Feature, abs(Coefficient))
  )

ggplot(coef_df, aes(x = abs(Coefficient), y = Feature, fill = Direction)) +
  geom_col(width = 0.7) +
  geom_text(
    aes(label = round(abs(Coefficient), 3)),
    hjust = -0.15,
    size  = 3.5,
    color = "grey30"
  ) +
  scale_x_continuous(expand = expansion(mult = c(0, 0.15))) +  # add right margin for labels
  geom_vline(xintercept = 0, linewidth = 0.4, color = "grey40") +
  scale_fill_manual(
    values = c("Positive" = "#2196F3", "Negative" = "#E53935"),
    name   = "Effect on\nConversion"
  ) +
  labs(
    title    = "Elastic-Net Feature Importance",
    subtitle = paste0("Non-zero coefficients at \u03bb = ", round(best_lambda, 4),
                      "  (\u03b1 = ", round(best_alpha, 4), ")"),
    x        = "Absolute Coefficient Value",
    y        = NULL
  ) +
  theme_minimal(base_size = 13) +
  theme(
    plot.title         = element_text(face = "bold", size = 15),
    plot.subtitle      = element_text(color = "grey50", size = 11),
    legend.position    = "right",
    panel.grid.major.y = element_blank(),
    panel.grid.minor   = element_blank()
  )

ggsave("plots/plots logistic regression/feature_importance.png", width = 8, height = 6, dpi = 150)
