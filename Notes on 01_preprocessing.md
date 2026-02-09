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



source("01\_preprocessing.R")



Make sure these utility files exist (they are sourced internally):



utils/utils\_preprocessing.R

utils/utils.R



##### **Function signature (what you can pass in)**



preprocess\_data(

&nbsp; data,

&nbsp; target\_col,

&nbsp; id\_cols = NULL,

&nbsp; filter\_col = NULL,

&nbsp; filter\_min = NULL,

&nbsp; filter\_max = NULL,

&nbsp; train\_prop = 0.8,

&nbsp; stratify = TRUE,

&nbsp; seed = 123,

&nbsp; model\_type = c("glmnet", "rf", "xgb")

)



###### **Required arguments**



1. data – data frame to preprocess
2. target\_col – name of the target variable (string)



Optional arguments



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



prep <- preprocess\_data(

&nbsp; data        = marketing\_data,

&nbsp; target\_col = "Conversion",

&nbsp; id\_cols    = "CustomerID",

&nbsp; filter\_col = "ConversionRate",

&nbsp; filter\_min = 0,

&nbsp; filter\_max = 1,

&nbsp; model\_type = "glmnet"

)



##### **How to access outputs**



The function returns a list.



X\_train <- prep$X\_train

y\_train <- prep$y\_train



X\_test  <- prep$X\_test

y\_test  <- prep$y\_test



obs\_weights <- prep$obs\_weights



You can inspect everything with:



names(prep)

str(prep)



##### **Model-specific behavior (important)**



Model type	Encoding	Scaling		Matrix output

glmnet		  Yes		  Yes		   Yes

rf		  Yes		  No		   No

xgb		  Yes		  No		   Yes





Outlier handling is always applied, regardless of model.



##### **Common mistakes to avoid**



❌ Forgetting to pass filter\_col (filter won’t run)

❌ Assuming scaling happens for tree models

❌ Modifying preprocessing outside the function

❌ Refitting preprocessors on test data



##### One-sentence summary



Call preprocess\_data() once per model, then extract X\_train, y\_train, etc. from the returned list and fit your model.

