#-----------------------------------
# Load libraries and preprocessing
#-----------------------------------

library(dplyr)
library(glmnet)
library(caret)

source("utils_preprocessing.R")
source("utils.R")

#--------------------------------
# Change the working directory
#   to where the file is located
#--------------------------------

set_script_wd()

#---------------------------------------
# Read data
#   Note: Make sure that the file is in 
#     the same directory as the script
#---------------------------------------

marketing_data <- read.csv("marketing.csv")

#----------------------
# Drop duplicate rows
#----------------------

marketing_data <- drop_duplicates(marketing_data)

#-----------------------
# Drop the Customer ID
#-----------------------

marketing_data <- marketing_data %>% 
  select(-CustomerID)

#----------------------------
# Filter out invalid numbers
#----------------------------

marketing_data <- marketing_data %>%
  filter(ConversionRate >= 0, ConversionRate <= 1)

#-------------
# Split data
#-------------

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