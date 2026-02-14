#-----------------------------------
# Load libraries and utilities
#-----------------------------------

library(dplyr)
library(glmnet)
library(caret)

source("utils/utils.R")
source("01_preprocessing.R")
#--------------------------------
# Change the working directory
#   to where the file is located
#--------------------------------

set_script_wd()

marketing_data <- read.csv("marketing.csv")

prep_glmnet <- preprocess_data(
  data        = marketing_data,
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

full_preprocessed_data <- prep_glmnet$full_processed_data