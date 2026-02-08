#-----------------------------------
# Load libraries and preprocessing
#-----------------------------------

library(dplyr)
library(glmnet)
library(caret)

source("utils_preprocessing.R")

#----------------
# Read data
#----------------

marketing_data <- read.csv("marketing.csv")

marketing_data <- marketing_data %>% 
  select(-CustomerID)
#----------------------------
# Split data
#----------------------------

split <- split_data(
  marketing_data,
  target = "Conversion",
  prop = 0.8,
  stratify = TRUE,
  seed = 123
)

train_data <- split$train
test_data  <- split$test

X_train <- train_data %>% select(-Conversion)
y_train <- as.factor(train_data$Conversion)

X_test  <- test_data %>% select(-Conversion)
y_test  <- as.factor(test_data$Conversion)

#---------------------------------
# Preprocessing 
#   (fit on train, apply to both)
#---------------------------------

# Impute Missing Values
imputer <- fit_imputer(X_train)
X_train <- apply_imputer(X_train, imputer)
X_test  <- apply_imputer(X_test,  imputer)

# Get rid of Outliers
outlier_handler <- fit_outlier_handler(X_train)
X_train <- apply_outlier_handler(X_train, outlier_handler)
X_test  <- apply_outlier_handler(X_test,  outlier_handler)

# Encode Categorical Variables
encoder <- fit_encoder(X_train)
X_train <- apply_encoder(X_train, encoder)
X_test  <- apply_encoder(X_test,  encoder)

# Fit and Apply a Zero Variance Filter
zv_filter <- fit_zv_filter(X_train)
X_train <- apply_zv_filter(X_train, zv_filter)
X_test  <- apply_zv_filter(X_test,  zv_filter)

# Fit and Apply a Z-score Scaler (Mean: 0, Std. Dev.: 1)
scaler <- fit_scaler(X_train)
X_train <- apply_scaler(X_train, scaler)
X_test  <- apply_scaler(X_test,  scaler)

# Convert Training and Test Data to Necessary Format
X_train <- as.matrix(X_train)
X_test  <- as.matrix(X_test)

#-----------------------------------
# Class weights
#   Minority class gets more weight
#   Majority class gets less weight
#-----------------------------------

class_weights <- compute_class_weights(y_train)
obs_weights   <- class_weights[as.character(y_train)]

#-----------------------------------
# Elastic-net CV (alpha + lambda)
#-----------------------------------

alpha_grid <- seq(0, 1, by = 0.1)
set.seed(123)

cv_models <- lapply(alpha_grid, function(a) {
  cv.glmnet(
    x = X_train,
    y = y_train,
    family = "binomial",
    alpha = a,
    weights = obs_weights,
    standardize = FALSE,
    keep = TRUE
  )
})

cv_errors <- sapply(cv_models, function(m) min(m$cvm))
best_idx  <- which.min(cv_errors)

best_alpha  <- alpha_grid[best_idx]
best_lambda <- cv_models[[best_idx]]$lambda.min
best_cv <- cv_models[[best_idx]]

lambda_idx <- which(best_cv$lambda == best_lambda)

cv_probs <- best_cv$fit.preval[, lambda_idx]
cv_probs <- as.numeric(cv_probs)

#-----------------------------------
# Threshold Tuning
#-----------------------------------

tune_threshold <- function(probs, y, grid = seq(0.05, 0.95, by = 0.01)) {
  scores <- sapply(grid, function(t) {
    preds <- factor(ifelse(probs > t, "1", "0"), levels = levels(y))
    confusionMatrix(preds, y, positive = "1")$byClass["Balanced Accuracy"]
  })
  grid[which.max(scores)]
}

best_threshold <- tune_threshold(cv_probs, y_train)

#-----------------------------------
# Final model
#-----------------------------------

final_model <- glmnet(
  x = X_train,
  y = y_train,
  family = "binomial",
  alpha = best_alpha,
  lambda = best_lambda,
  weights = obs_weights,
  standardize = FALSE
)

#-----------------------------------
# Test evaluation (ONLY here)
#-----------------------------------

prob_test <- predict(final_model, X_test, type = "response")

pred_test <- factor(
  ifelse(prob_test > best_threshold, "1", "0"),
  levels = levels(y_train)
)

cm <- confusionMatrix(
  pred_test,
  y_test,
  positive = "1"
)

print(cm)

print(best_threshold)
