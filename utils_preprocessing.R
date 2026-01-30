library(dplyr)

impute_by_group_median <- function(data, group_col, add_indicators = TRUE) {
  
  data <- data %>%
    mutate(across(
      everything(),
      ~ .,
      .names = "{.col}"
    ))
  
  # Add missingness indicators if requested
  if (add_indicators) {
    data <- data %>%
      mutate(across(
        -all_of(group_col),
        ~ ifelse(is.na(.), 1, 0),
        .names = "{.col}_missing"
      ))
  }
  
  # Impute numeric columns by group-specific median
  data <- data %>%
    group_by(across(all_of(group_col))) %>%
    mutate(across(
      where(is.numeric),
      ~ ifelse(is.na(.), median(., na.rm = TRUE), .)
    )) %>%
    ungroup()
  
  return(data)
}

remove_outliers_iqr <- function(data, multiplier = 1.5) {
  
  # Identify numeric columns
  numeric_cols <- sapply(data, is.numeric)
  
  # If no numeric columns, return data unchanged
  if (!any(numeric_cols)) {
    warning("No numeric columns found. Returning original data.")
    return(data)
  }
  
  # Initialize logical vector to track outlier rows
  outlier_rows <- rep(FALSE, nrow(data))
  
  # Loop over numeric columns
  for (col in names(data)[numeric_cols]) {
    
    x <- data[[col]]
    
    # Skip columns with all NA or zero variance
    if (all(is.na(x)) || length(unique(x[!is.na(x)])) <= 1) {
      next
    }
    
    Q1 <- quantile(x, 0.25, na.rm = TRUE)
    Q3 <- quantile(x, 0.75, na.rm = TRUE)
    IQR_val <- Q3 - Q1
    
    lower <- Q1 - multiplier * IQR_val
    upper <- Q3 + multiplier * IQR_val
    
    outlier_rows <- outlier_rows | (x < lower | x > upper)
  }
  
  # Remove rows with any outlier
  cleaned_data <- data[!outlier_rows, ]
  
  return(cleaned_data)
  
}

one_hot_encode <- function(df, drop_first = FALSE) {
  # Identify categorical columns (factor or character)
  cat_cols <- names(df)[sapply(df, function(x) is.factor(x) || is.character(x))]
  
  # If no categorical variables, return original df
  if (length(cat_cols) == 0) {
    return(df)
  }
  
  # Convert characters to factors
  df[cat_cols] <- lapply(df[cat_cols], as.factor)
  
  # Build formula dynamically
  if (drop_first) {
    formula <- as.formula(paste("~", paste(cat_cols, collapse = " + "), "- 1"))
  } else {
    formula <- as.formula(paste("~", paste(cat_cols, collapse = " + ")))
  }
  
  # Create design matrix
  dummies <- model.matrix(formula, data = df)
  
  # Remove intercept if present
  dummies <- dummies[, colnames(dummies) != "(Intercept)", drop = FALSE]
  
  # Combine with non-categorical columns
  num_df <- df[ , !(names(df) %in% cat_cols), drop = FALSE]
  
  final_df <- cbind(num_df, dummies)
  
  return(as.data.frame(final_df))
}
