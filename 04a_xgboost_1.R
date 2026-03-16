# 04a_xgboost_sample.R — XGBoost Predictive Model for Customer Conversion
# LZSCC.460 Group Project - Objective 4
#
# Imbalance strategy: observation weights from preprocessing

# This file contains the full pipeline for building an XGBoost model
# to predict whether a customer will convert or not
# The model uses observation weights to handle class imbalance
# The pipeline goes through: preprocessing, tuning, training, evaluation,
# feature importance, and SHAP analysis


# Loading all the libraries we need for this model
# Each one handles a different part of the pipeline

library(xgboost)        # The main XGBoost library that builds the gradient boosted trees
library(dplyr)          # For data manipulation like filtering and selecting columns
library(caret)          # For the confusionMatrix function that calculates all our metrics
library(pROC)           # For computing AUC-ROC which measures overall ranking ability
library(PRROC)          # For computing PR-AUC which is better for imbalanced datasets
library(ggplot2)        # For creating all our plots like feature importance
library(SHAPforxgboost) # For computing SHAP values that explain individual predictions


# Setting the working directory so R knows where to find our files
# All file paths after this will be relative to this folder
setwd("C:/Users/sejdi/Desktop/Lancaster University Masters - Leipzig/Data Science Fundamentals/LZSCC.460-Group-Project")

# Loading our utility functions and preprocessing pipeline
# utils_preprocessing.R has all the individual preprocessing steps like imputation and encoding
# utils.R has helper functions like set_script_wd()
# 01_preprocessing.R has the main preprocess_data() function that runs the full pipeline
source("utils/utils_preprocessing.R")
source("utils/utils.R")
source("01_preprocessing.R")

# Creating the output folder for all XGBoost plots and results
# recursive = TRUE means it will create both plots/ and plots/xgboost/ if needed
if (!dir.exists("plots/xgboost")) dir.create("plots/xgboost", recursive = TRUE)


# Loading the raw dataset and running it through preprocessing
# preprocess_data() handles everything: duplicate removal, dropping CustomerID,
# filtering ConversionRate to valid 0-1 range, stratified 80/20 train/test split,
# gender-stratified median imputation for missing values, IQR outlier capping,
# one-hot encoding of categorical variables, matrix conversion for XGBoost,
# and computing inverse-frequency class weights
# model_type = "xgb" means no scaling is applied (trees dont need it)
# and the output is converted to matrix format which XGBoost requires
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

# Extracting the pieces we need from the preprocessing output
# X_train and X_test are the feature matrices (everything except Conversion)
# y_train and y_test are the target vectors (just 0s and 1s)
# obs_weights are the inverse-frequency weights that give non-converters higher importance
# We convert y_train and y_test to numeric because XGBoost needs numbers not factors
X_train     <- processed$X_train
y_train     <- as.numeric(as.character(processed$y_train))
X_test      <- processed$X_test
y_test      <- as.numeric(as.character(processed$y_test))
obs_weights <- processed$obs_weights

# Printing the dimensions so we can verify the split worked correctly
print(dim(X_train))
print(dim(X_test))


# Counting how many converters and non-converters we have in the training set
# This helps us understand the severity of the imbalance
# n_pos is the number of converters (class 1) which should be around 88%
# n_neg is the number of non-converters (class 0) which should be around 12%
n_pos <- sum(y_train == 1)
n_neg <- sum(y_train == 0)

print(n_pos)
print(n_neg)

# Creating DMatrix objects which is XGBoost's internal data format
# DMatrix stores the features, labels, and weights together in a compressed way
# This makes training much faster compared to using regular data frames
# The training DMatrix includes observation weights so the model knows
# to penalise misclassifying non-converters more heavily
# The test DMatrix does not need weights because we only use it for evaluation
dtrain <- xgb.DMatrix(data = X_train, label = y_train, weight = obs_weights)
dtest  <- xgb.DMatrix(data = X_test,  label = y_test)


# Setting up the hyperparameter search space
# We are going to try 50 random combinations of 6 different settings
# set.seed(42) makes sure the random combinations are the same every time we run this
# This is important for reproducibility - anyone running this code gets the same results
set.seed(42)
n_iter <- 50

# Each row of search_space is one random combination of hyperparameters
# runif generates random decimal numbers between the two limits
# sample generates random whole numbers from the given range
# These ranges were chosen based on commonly recommended values
search_space <- data.frame(
  eta              = runif(n_iter, 0.01, 0.3),
  max_depth        = sample(3:10, n_iter, replace = TRUE),
  min_child_weight = sample(1:10, n_iter, replace = TRUE),
  subsample        = runif(n_iter, 0.5, 1.0),
  colsample_bytree = runif(n_iter, 0.5, 1.0),
  gamma            = runif(n_iter, 0, 5)
)

# This function runs the entire random search tuning process
# It takes the training data and the search space
# For each of the 50 combinations it runs 5-fold cross-validation
# and records how well that combination performed
# At the end it returns the best combination (lowest cross-validated log-loss)
run_tuning <- function(dtrain, search_space) {
  
  # Setting up an empty results table to store the outcome of each iteration
  # Each row will hold the hyperparameters used and the resulting performance
  n <- nrow(search_space)
  results <- data.frame(
    iter = 1:n, eta = numeric(n), max_depth = integer(n),
    min_child_weight = integer(n), subsample = numeric(n),
    colsample_bytree = numeric(n), gamma = numeric(n),
    best_nrounds = integer(n), cv_logloss = numeric(n)
  )
  
  # Looping through each of the 50 random combinations
  for (i in seq_len(n)) {
    
    # Building the parameter list for this specific combination
    # objective = "binary:logistic" tells XGBoost this is a binary classification task
    # eval_metric = "logloss" tells it to measure performance using log-loss
    # nthread uses all available CPU cores minus one so the computer stays responsive
    params_i <- list(
      objective = "binary:logistic", eval_metric = "logloss",
      eta = search_space$eta[i], max_depth = search_space$max_depth[i],
      min_child_weight = search_space$min_child_weight[i],
      subsample = search_space$subsample[i],
      colsample_bytree = search_space$colsample_bytree[i],
      gamma = search_space$gamma[i],
      nthread = parallel::detectCores() - 1
    )
    
    # Running 5-fold stratified cross-validation with these parameters
    # set.seed(123) ensures the same fold splits every time for fair comparison
    # nrounds = 1000 allows up to 1000 trees to be built
    # nfold = 5 splits the training data into 5 equal parts
    # stratified = TRUE keeps the same class balance in each fold
    # early_stopping_rounds = 30 means if log-loss doesnt improve for 30 rounds
    # in a row, we stop building more trees - this prevents overfitting
    # verbose = 0 stops it from printing progress for each round
    set.seed(123)
    cv <- xgb.cv(params = params_i, data = dtrain, nrounds = 1000,
                 nfold = 5, stratified = TRUE, early_stopping_rounds = 30,
                 verbose = 0)
    
    # Getting the best round number (where log-loss was lowest before early stopping)
    iter_best <- cv$early_stop$best_iteration
    
    # Finding the column name that contains the test log-loss mean
    # XGBoost names this column differently across versions so we search for it
    log_col   <- grep("test.*logloss_mean", names(cv$evaluation_log), value = TRUE)[1]
    
    # Storing all the results for this iteration in our results table
    results[i, ] <- list(i, search_space$eta[i], search_space$max_depth[i],
                         search_space$min_child_weight[i], search_space$subsample[i],
                         search_space$colsample_bytree[i], search_space$gamma[i],
                         iter_best, cv$evaluation_log[[log_col]][iter_best])
  }
  
  # After all 50 iterations, we find which one had the lowest log-loss
  # That combination becomes our chosen set of hyperparameters
  best <- results[which.min(results$cv_logloss), ]
  list(results = results, best = best)
}

# Running the tuning process and storing the results
tune <- run_tuning(dtrain, search_space)

# Printing the best hyperparameters found
print(tune$best)


# Building the final model using the best hyperparameters found during tuning
# We take each setting from tune$best and put them into the params list
# This is the model that will be evaluated on the test set
params <- list(
  objective = "binary:logistic", eval_metric = "logloss",
  eta = tune$best$eta, max_depth = tune$best$max_depth,
  min_child_weight = tune$best$min_child_weight,
  subsample = tune$best$subsample,
  colsample_bytree = tune$best$colsample_bytree,
  gamma = tune$best$gamma,
  nthread = parallel::detectCores() - 1
)

# Training the final model on the full training set
# nrounds is set to the best number of rounds found during tuning
# evals tracks both training and test performance during training
# This is just for monitoring - the test set is NOT used for any training decisions
# verbose = 0 keeps the output clean
model <- xgb.train(params = params, data = dtrain,
                   nrounds = tune$best$best_nrounds,
                   evals = list(train = dtrain, test = dtest), verbose = 0)


# This function evaluates a trained model on the test set
# It calculates all the metrics we care about
# It takes the model, the test DMatrix, and the actual labels
evaluate <- function(model, dtest, y_actual) {
  
  # Getting predicted probabilities for each customer in the test set
  # These are numbers between 0 and 1 representing how likely conversion is
  probs <- predict(model, dtest)
  
  # Converting probabilities to binary predictions using 0.5 as the cutoff
  # If the probability is above 0.5, we predict "convert" (1)
  # If below 0.5, we predict "not convert" (0)
  preds <- ifelse(probs > 0.5, 1, 0)
  
  # Creating a confusion matrix which compares our predictions to the actual outcomes
  # This gives us sensitivity, specificity, accuracy, balanced accuracy, F1, etc.
  # positive = "1" tells it that class 1 (converter) is the positive class
  cm  <- confusionMatrix(factor(preds, levels = c("0","1")),
                         factor(y_actual, levels = c("0","1")), positive = "1")
  
  # Computing the ROC curve and AUC-ROC
  # AUC-ROC measures how well the model ranks converters above non-converters
  # A perfect model would score 1.0, random guessing would score 0.5
  roc <- roc(y_actual, probs, quiet = TRUE)
  
  # Computing PR-AUC
  # PR-AUC is more informative than AUC-ROC when the data is imbalanced
  # scores.class0 gets the probabilities for the positive class (converters)
  # scores.class1 gets the probabilities for the negative class (non-converters)
  pr <- pr.curve(scores.class0 = probs[y_actual == 1],
                 scores.class1 = probs[y_actual == 0])
  
  # Returning everything so we can use it later
  list(probs = probs, cm = cm, roc = roc, pr = pr)
}

# Running the evaluation on our held-out test set
# This is the first and only time the test data is used
eval_result <- evaluate(model, dtest, y_test)

# Storing all the metrics in a clean data frame
# as.numeric() is needed because some metrics come wrapped in named vectors
# round() keeps everything to 4 decimal places for readability
# We save this as a CSV so we can compare it with other models later
metrics <- data.frame(
  Model        = "XGBoost",
  AUC_ROC      = round(as.numeric(auc(eval_result$roc)), 4),
  PR_AUC       = round(as.numeric(eval_result$pr$auc.integral), 4),
  Accuracy     = round(as.numeric(eval_result$cm$overall["Accuracy"]), 4),
  Balanced_Acc = round(as.numeric(eval_result$cm$byClass["Balanced Accuracy"]), 4),
  Sensitivity  = round(as.numeric(eval_result$cm$byClass["Sensitivity"]), 4),
  Specificity  = round(as.numeric(eval_result$cm$byClass["Specificity"]), 4),
  F1           = round(as.numeric(eval_result$cm$byClass["F1"]), 4)
)

print(metrics)
print(eval_result$cm$table)

write.csv(metrics, "plots/xgboost/metrics_sample.csv", row.names = FALSE)


# Extracting gain-based feature importance from the trained model
# Gain measures how much each feature improved predictions across all tree splits
# Every time the model splits the data on a feature, the prediction gets a bit better
# The total improvement from each feature across all trees is summed up
# Features with higher total gain are considered more important
imp <- xgb.importance(feature_names = colnames(X_train), model = model)

print(head(imp, 10))

# Taking only the top 15 features for the plot
# We reverse the factor levels so the most important feature appears at the top
# coord_flip() makes it a horizontal bar chart which is easier to read with long names
imp_top <- imp[1:min(15, nrow(imp)), ]
imp_top$Feature <- factor(imp_top$Feature, levels = rev(imp_top$Feature))

p_imp <- ggplot(imp_top, aes(x = Feature, y = Gain)) +
  geom_col(fill = "#2563eb") + coord_flip() +
  labs(title = "Feature Importance - XGBoost", x = NULL, y = "Gain") +
  theme_minimal()
ggsave("plots/xgboost/importance_sample.png", p_imp, width = 10, height = 6, dpi = 150)


# Computing SHAP values for every customer in the training set
# SHAP tells us how much each feature pushed each individual prediction
# higher or lower - unlike gain which just tells us overall importance
# For example, SHAP might say "this customer's high TimeOnSite pushed
# their conversion probability up by 0.15"
# shap.values() computes the raw SHAP scores for every feature and every customer
# shap.prep() reshapes them into long format needed for the summary plot
shap <- shap.values(xgb_model = model, X_train = X_train)
shap_long <- shap.prep(shap_contrib = shap$shap_score,
                       X_train = as.data.frame(X_train))

# Creating the SHAP beeswarm summary plot
# Each dot is one customer, positioned by how much that feature affected their prediction
# Dots to the right mean the feature pushed toward conversion
# Dots to the left mean it pushed away from conversion
# The colour shows whether the actual feature value was high (purple) or low (yellow)
p_shap <- shap.plot.summary(shap_long) +
  ggtitle("SHAP Summary - XGBoost") + theme_minimal()
ggsave("plots/xgboost/shap_sample.png", p_shap, width = 10, height = 7, dpi = 150)

# Printing the top 10 features ranked by their average absolute SHAP value
# This is another way to rank features - by average impact on predictions
print(head(sort(shap$mean_shap_score, decreasing = TRUE), 10))
