# **📦 Preprocessing Guide (01\_preprocessing.R)**



This file defines a single entry-point function, preprocess\_data(), which prepares a dataset for multiple model types in a consistent and reusable way.



It replaces the old script-style preprocessing with a callable function.



##### **What preprocess\_data() does**



When called, the function:



1. Removes duplicate rows
2. Optionally removes ID columns
3. Optionally filters invalid values in a specified column
4. Splits data into train / test sets
5. Fits preprocessing steps on training data only:

&nbsp;	- missing-value imputation

&nbsp;	- outlier handling

&nbsp;	- categorical encoding

&nbsp;	- (optional) feature scaling

6\. Applies the same preprocessing to test data

7\. Returns ready-to-use training and test sets



All preprocessing logic lives inside this function.



##### **How to load it**



In your modeling script, do:


```r
source("01_preprocessing.R")
```


Make sure these utility files exist (they are sourced internally):


```r
utils/utils_preprocessing.R

utils/utils.R
```


##### **Function signature (what you can pass in)**


```r
preprocess_data(
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
)
```


###### **Required arguments**

1. data – data frame to preprocess
2. target\_col – name of the target variable (string)



###### **Optional arguments**

1. id\_cols – ID columns to drop (string or character vector)
2. filter\_col – column to apply value filtering on
3. filter\_min / filter\_max – numeric bounds for filtering
4. train\_prop – train/test split ratio (default 80/20)
5. stratify – whether to stratify by target (default TRUE)
6. seed – random seed for reproducibility
7. model\_type – model the data is prepared for:

&nbsp;	- "glmnet" → elastic-net logistic regression

&nbsp;	- "rf" → random forest

&nbsp;	- "xgb" → XGBoost



##### **Most common usage (example)**

Equivalent to the old preprocessing script:

```r
prep <- preprocess\_data(
  data = marketing_data,
  target_col = "Conversion",
  id_cols    = "CustomerID",
  filter_col = "ConversionRate",
  filter_min = 0,
  filter_max = 1,
  model_type = "glmnet"
)
```

##### **How to access outputs**

The function returns a list.

```r
X_train <- prep$X_train
y_train <- prep$y_train

X_test  <- prep$X_test
y_test  <- prep$y_test

obs_weights <- prep$obs_weights
```

You can inspect everything with:

```r
names(prep)

str(prep)
```

##### **Model-specific behavior (important)**

| Model type | Encoding | Scaling | Matrix output |
|------------|----------|---------|---------------|
| glmnet		 | Yes		  | Yes		  | Yes           |
| rf		     | Yes		  | No		  | No            |
| xgb		     | Yes		  | No		  | Yes           |

Outlier handling is always applied, regardless of model.

##### **Common mistakes to avoid**

❌ Forgetting to pass filter\_col (filter won’t run)
❌ Assuming scaling happens for tree models
❌ Modifying preprocessing outside the function
❌ Refitting preprocessors on test data

##### One-sentence summary

Call `preprocess_data()` once per model, then extract `X_train`, `y_train`, etc. from the returned list and fit your model.

