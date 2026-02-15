preprocess_data <- function(
    data,
    target_col,
    id_cols = NULL,
    filter_col = NULL,
    filter_min = NULL,
    filter_max = NULL,
    train_prop = 0.8,
    stratify = TRUE,
    seed = 123,
    model_type = c("glmnet", "rf", "xgb")
) {
  
  model_type <- match.arg(model_type)
  
  library(dplyr)
  library(glmnet)
  library(caret)
  
  source("utils/utils_preprocessing.R")
  source("utils/utils.R")
  
  #----------------------
  # Drop duplicate rows
  #----------------------
  data <- drop_duplicates(data)
  
  #-----------------------
  # Drop ID columns
  #-----------------------
  if (!is.null(id_cols)) {
    data <- data %>% select(-all_of(id_cols))
  }
  
  #----------------------------
  # Generic filtering step
  #----------------------------
  if (!is.null(filter_col)) {
    if (!is.null(filter_min)) {
      data <- data %>% filter(.data[[filter_col]] >= filter_min)
    }
    if (!is.null(filter_max)) {
      data <- data %>% filter(.data[[filter_col]] <= filter_max)
    }
  }
  
  #-------------
  # Split data
  #-------------
  split <- split_data(
    data,
    target = target_col,
    prop = train_prop,
    stratify = stratify,
    seed = seed
  )
  
  train_data <- split$train
  test_data  <- split$test
  
  X_train <- train_data %>% select(-all_of(target_col))
  y_train <- as.factor(train_data[[target_col]])
  
  X_test  <- test_data %>% select(-all_of(target_col))
  y_test  <- as.factor(test_data[[target_col]])
  
  #---------------------------------
  # Preprocessing (fit on TRAIN)
  #---------------------------------
  
  # Imputation
  imputer <- fit_imputer(X_train)
  X_train <- apply_imputer(X_train, imputer)
  X_test  <- apply_imputer(X_test,  imputer)
  
  # Outlier handling
  outlier_handler <- fit_outlier_handler(X_train)
  X_train <- apply_outlier_handler(X_train, outlier_handler)
  X_test  <- apply_outlier_handler(X_test,  outlier_handler)
  
  #-----------------------------------
  # Create FULL dataset for EDA
  # (Before encoding & scaling)
  #-----------------------------------
  
  X_full <- data %>% select(-all_of(target_col))
  y_full <- as.factor(data[[target_col]])
  
  X_full <- apply_imputer(X_full, imputer)
  X_full <- apply_outlier_handler(X_full, outlier_handler)
  
  full_processed_data <- X_full
  full_processed_data[[target_col]] <- y_full
  
  #---------------------------------
  # Continue MODEL preprocessing
  #---------------------------------
  
  # Encoding (for modeling only)
  encoder <- fit_encoder(X_train)
  X_train <- apply_encoder(X_train, encoder)
  X_test  <- apply_encoder(X_test,  encoder)
  
  # Scaling (glmnet only)
  if (model_type == "glmnet") {
    scaler <- fit_scaler(X_train)
    X_train <- apply_scaler(X_train, scaler)
    X_test  <- apply_scaler(X_test,  scaler)
  } else {
    scaler <- NULL
  }
  
  # Matrix conversion
  if (model_type %in% c("glmnet", "xgb")) {
    X_train <- as.matrix(X_train)
    X_test  <- as.matrix(X_test)
  }
  
  #-----------------------------------
  # Class weights
  #-----------------------------------
  
  class_weights <- compute_class_weights(y_train)
  obs_weights   <- class_weights[as.character(y_train)]
  
  #----------------------
  # Return results
  #----------------------
  
  list(
    X_train = X_train,
    y_train = y_train,
    X_test  = X_test,
    y_test  = y_test,
    obs_weights = obs_weights,
    model_type = model_type,
    preprocessors = list(
      imputer = imputer,
      outlier_handler = outlier_handler,
      encoder = encoder,
      scaler = scaler
    ),
    full_processed_data = full_processed_data
  )
}