library(dplyr)
library(caret)

split_data <- function(df, target, prop = 0.8, stratify = TRUE, seed = NULL) {
  # df: data.frame containing predictors + target
  # target: name of target column (string)
  # prop: proportion of data used for training
  # stratify: whether to stratify by target
  # seed: optional random seed
  
  if (!is.null(seed)) set.seed(seed)
  
  y <- df[[target]]
  n <- nrow(df)
  
  if (stratify) {
    y <- as.factor(y)
    idx <- unlist(
      lapply(levels(y), function(lvl) {
        lvl_idx <- which(y == lvl)
        sample(lvl_idx, floor(prop * length(lvl_idx)))
      })
    )
  } else {
    idx <- sample(seq_len(n), floor(prop * n))
  }
  
  train <- df[idx, , drop = FALSE]
  test  <- df[-idx, , drop = FALSE]
  
  list(
    train = train,
    test = test
  )
}

fit_outlier_handler <- function(df, k = 1.5) {
  # Select numeric columns only
  num_cols <- names(df)[sapply(df, is.numeric)]
  
  bounds <- lapply(num_cols, function(col) {
    x <- df[[col]]
    x <- x[!is.na(x)]
    
    q1 <- quantile(x, 0.25)
    q3 <- quantile(x, 0.75)
    iqr <- q3 - q1
    
    list(
      lower = q1 - k * iqr,
      upper = q3 + k * iqr
    )
  })
  
  names(bounds) <- num_cols
  
  return(bounds)
}

apply_outlier_handler <- function(df, bounds) {
  for (col in names(bounds)) {
    if (!col %in% names(df)) next
    
    lower <- bounds[[col]]$lower
    upper <- bounds[[col]]$upper
    
    df[[col]] <- pmin(pmax(df[[col]], lower), upper)
  }
  
  return(df)
}

fit_imputer <- function(df, gender_col = "gender") {
  # Ensure gender is factor
  df[[gender_col]] <- as.factor(df[[gender_col]])
  
  # Identify numeric columns (exclude gender)
  num_cols <- names(df)[sapply(df, is.numeric)]
  num_cols <- setdiff(num_cols, gender_col)
  
  # Compute gender-specific medians
  gender_medians <- lapply(num_cols, function(col) {
    tapply(df[[col]], df[[gender_col]], median, na.rm = TRUE)
  })
  
  names(gender_medians) <- num_cols
  
  # Global medians as fallback
  global_medians <- sapply(df[num_cols], median, na.rm = TRUE)
  
  list(
    gender_col = gender_col,
    num_cols = num_cols,
    gender_medians = gender_medians,
    global_medians = global_medians
  )
}

apply_imputer <- function(df, imputer) {
  gender_col <- imputer$gender_col
  
  # Ensure gender is factor
  df[[gender_col]] <- as.factor(df[[gender_col]])
  
  for (col in imputer$num_cols) {
    missing_idx <- is.na(df[[col]])
    
    if (any(missing_idx)) {
      genders <- as.character(df[[gender_col]][missing_idx])
      
      imputed_values <- mapply(
        function(g) {
          if (!is.na(imputer$gender_medians[[col]][g])) {
            imputer$gender_medians[[col]][g]
          } else {
            imputer$global_medians[col]
          }
        },
        genders
      )
      
      df[[col]][missing_idx] <- imputed_values
    }
  }
  
  df
}

fit_encoder <- function(df) {
  # Identify categorical columns
  cat_cols <- names(df)[sapply(df, function(x) is.factor(x) || is.character(x))]
  
  # Convert character columns to factors
  df[cat_cols] <- lapply(df[cat_cols], as.factor)
  
  # Build formula for model.matrix
  if (length(cat_cols) > 0) {
    formula <- as.formula(paste("~", paste(cat_cols, collapse = " + ")))
    design_mat <- model.matrix(formula, data = df)
    
    # Remove intercept
    design_mat <- design_mat[, colnames(design_mat) != "(Intercept)", drop = FALSE]
    
    dummy_cols <- colnames(design_mat)
  } else {
    dummy_cols <- character(0)
  }
  
  list(
    cat_cols   = cat_cols,
    dummy_cols = dummy_cols
  )
}

apply_encoder <- function(df, encoder) {
  # Convert character columns to factors using training levels
  for (col in encoder$cat_cols) {
    df[[col]] <- factor(df[[col]], levels = levels(df[[col]]))
  }
  
  # Generate dummy variables
  if (length(encoder$cat_cols) > 0) {
    formula <- as.formula(paste("~", paste(encoder$cat_cols, collapse = " + ")))
    dummies <- model.matrix(formula, data = df)
    
    # Remove intercept
    dummies <- dummies[, colnames(dummies) != "(Intercept)", drop = FALSE]
    
    # Ensure same dummy columns as training
    missing_cols <- setdiff(encoder$dummy_cols, colnames(dummies))
    if (length(missing_cols) > 0) {
      dummies <- cbind(dummies,
                       matrix(0, nrow = nrow(dummies), ncol = length(missing_cols),
                              dimnames = list(NULL, missing_cols)))
    }
    
    # Drop extra columns and reorder
    dummies <- dummies[, encoder$dummy_cols, drop = FALSE]
  }
  
  # Remove original categorical columns
  num_df <- df[, !(names(df) %in% encoder$cat_cols), drop = FALSE]
  
  # Combine numeric + encoded categorical
  final_df <- cbind(num_df, dummies)
  
  as.data.frame(final_df)
}

fit_zv_filter <- function(df, freq_cutoff = 0.95) {
  # Keep only numeric columns
  num_df <- df[, sapply(df, is.numeric), drop = FALSE]
  
  keep_cols <- sapply(num_df, function(x) {
    if (var(x, na.rm = TRUE) == 0) {
      return(FALSE)
    }
    freq_ratio <- max(table(x)) / length(x)
    freq_ratio < freq_cutoff
  })
  
  list(
    keep_columns = names(num_df)[keep_cols],
    removed_columns = names(num_df)[!keep_cols],
    freq_cutoff = freq_cutoff
  )
}

apply_zv_filter <- function(df, zv_object) {
  # Keep non-numeric columns untouched
  non_num <- df[, !sapply(df, is.numeric), drop = FALSE]
  
  # Keep only numeric columns seen during fitting
  num <- df[, intersect(zv_object$keep_columns, names(df)), drop = FALSE]
  
  # Recombine
  cbind(non_num, num)
}

fit_scaler <- function(df) {
  num_cols <- names(df)[sapply(df, is.numeric)]
  
  means <- sapply(df[num_cols], mean, na.rm = TRUE)
  sds   <- sapply(df[num_cols], sd, na.rm = TRUE)
  
  # Avoid division by zero
  sds[sds == 0] <- 1
  
  list(
    num_cols = num_cols,
    means = means,
    sds = sds
  )
}

apply_scaler <- function(df, scaler) {
  missing_cols <- setdiff(scaler$num_cols, names(df))
  if (length(missing_cols) > 0) {
    stop("apply_scaler(): missing columns: ", paste(missing_cols, collapse = ", "))
  }
  
  df_scaled <- df
  
  for (col in scaler$num_cols) {
    df_scaled[[col]] <- (df_scaled[[col]] - scaler$means[col]) / scaler$sds[col]
  }
  
  df_scaled
}

compute_class_weights <- function(y, normalize = TRUE) {
  # y: outcome vector (factor, character, or numeric)
  # normalize: if TRUE, mean weight = 1
  
  y <- as.factor(y)
  class_counts <- table(y)
  n_classes <- length(class_counts)
  n_obs <- length(y)
  
  # Inverse-frequency weights
  class_weights <- n_obs / (n_classes * class_counts)
  
  if (normalize) {
    class_weights <- class_weights / mean(class_weights)
  }
  
  # Return named vector
  as.numeric(class_weights) |> 
    setNames(names(class_counts))
}
