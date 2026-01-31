#-----------------------------------
# Load necessary libraries and files
#-----------------------------------

library(dplyr)
library(ggplot2)
library(tidyr)
library(caret)

source("utils_preprocessing.R")

#----------------
# Read data
#----------------

marketing_data <- read.csv(file.path(getwd(), "marketing.csv"), header = TRUE, sep = ",")

#----------------------------
# Split data
#----------------------------

split <- split_data(marketing_data, target="Conversion", prop=0.8, stratify = TRUE, seed = 123)

train_data <- split$train
test_data <- split$test

#------------------------
# Handle Outliers
#------------------------

outlier_handler <- fit_outlier_handler()