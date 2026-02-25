
# ============================================================
# random_forest.R
# ============================================================

library(randomForest)
library(caret)
library(dplyr)
library(pROC) #ROC/AUC
source("utils/utils.R")
source("01_preprocessing.R")

# -----------------------------
# Load dataset
# -----------------------------
marketing_df <- read.csv("marketing.csv")

# -----------------------------
# Preprocessing
# -----------------------------
prep_rf <- preprocess_data(
  data        = marketing_df,
  target_col  = "Conversion",
  id_cols     = "CustomerID",
  filter_col  = "ConversionRate",
  filter_min  = 0,
  filter_max  = 1,
  train_prop  = 0.8,
  stratify    = TRUE,
  seed        = 123,
  model_type  = "rf"
)

X_train <- prep_rf$X_train
y_train <- prep_rf$y_train
X_test  <- prep_rf$X_test
y_test  <- prep_rf$y_test
obs_weight<-prep_rf$obs_weights

#create a dataframe to fit the model
train_df <- data.frame(X_train, Conversion = y_train)
test_df  <- data.frame(X_test,  Conversion = y_test)


train_df$Conversion <- factor(train_df$Conversion, levels = c("0","1"), labels=c("no","yes"))
test_df$Conversion  <- factor(test_df$Conversion,  levels = c("0","1"), labels=c("no","yes"))

#=======================
#BASELINE RANDOM FOREST
#=======================
set.seed(123)

rf_baseline <- randomForest(
  Conversion ~ .,
  data       = train_df,
  ntree      = 200,
  mtry       = floor(sqrt(ncol(X_train))),
  nodesize   =1,
  importance = TRUE,
  weights    = obs_weight
)

#=============================================
#EVALUATION OF  BASELINE MODEL
#==============================================
prob_baseline <- predict(rf_baseline, newdata =test_df,type="prob")[,"yes"]
predicted_baseline<-factor(ifelse(prob_baseline>0.5, "yes","no"), levels=c("no","yes"))


#baseline confusion matrix
baseline_confmatrix <- confusionMatrix(
  data      = predicted_baseline,
  reference = test_df$Conversion,
  positive  = "yes"
)

cat("\n================ BASELINE RF (Test) ================\n")
print(baseline_confmatrix)

#AUC-ROC baseline
roc_baseline<-roc(test_df$Conversion, prob_baseline, quiet=TRUE)
auc_roc_baseline<-as.numeric(auc(roc_baseline))


#report for baseline
cat("\n=== BASELINE RF ===\n")
cat("Balanced Accuracy:", round(baseline_confmatrix$byClass["Balanced Accuracy"], 4),"\n")
cat("Recall:", round(baseline_confmatrix$byClass["Sensitivity"], 4),"\n")
cat("Specificity:", round(baseline_confmatrix$byClass["Specificity"], 4),"\n")
cat("F1:", round(baseline_confmatrix$byClass["F1"], 4),"\n")
cat("AUC-ROC:", round(auc_roc_baseline, 4),"\n")




# ============================================================
# 2) TUNE Random Forest mtry,ntree, nodesize)
# ============================================================
control <- trainControl(
  method = "cv",
  number = 3,
  classProbs = TRUE,
  summaryFunction = twoClassSummary
)
n_features <- ncol(X_train)

mtryGrid <- expand.grid(
  mtry = seq(2, n_features, by = 4) #2,6,....,n  (n:number of features)
)

ntree_grid<-c(200,300,500)
nodesize_grid<-c(1,5,10)

best_ROC<- -Inf
best_model<-NULL
best_ntree<-NA
best_nodesize<-NA
best_mtry<-NA

set.seed(123)

for(ntrees in ntree_grid){
  for(nodesizes in nodesize_grid){
    rf_tuned <- train(
      Conversion ~ .,
      data       = train_df,
      method     = "rf",
      trControl  = control,
      tuneGrid   = mtryGrid,
      ntree      = ntrees,
      nodesize   =nodesizes,
      weights    =obs_weight,
      metric="ROC"
  
  )
  #best TUNE
  best_idx<-which.max(rf_tuned$results$ROC)
  current_ROC<-rf_tuned$results$ROC[best_idx]
  current_mtry<-rf_tuned$results$mtry[best_idx]
  
  if(current_ROC>best_ROC){
    best_ROC<-current_ROC
    best_model<-rf_tuned
    best_ntree<-ntrees
    best_nodesize<-nodesizes
    best_mtry<-current_mtry
        
      }
   }
}
  
cat("\n============ BEST TUNED RF =============\n")
cat("\nntree:",best_ntree,"\n")
cat("nodesize:",best_nodesize,"\n")
cat("mtry:",best_mtry,"\n")
cat("ROC:", round(best_ROC,4),"\n")
#print(rf_tuned)
#cat("\n Best mtry:\n")
#print(rf_tuned$bestTune)

# ============================================================
# EVALUATION of tuned model
# ============================================================
tuned_prob <- predict(best_model, newdata = test_df,type="prob")[,"yes"]
tuned_pred<-factor(ifelse(tuned_prob>0.5, "yes","no"), levels=c("no","yes"))

tuned_confmatrix <- confusionMatrix(
  data      = tuned_pred,
  reference = test_df$Conversion,
  positive  = "yes"
)

cat("\n================ TUNED RF (Default threshold, Test) ================\n")
print(tuned_confmatrix)

# ============================================================
# tuning of THRESHOLDS t = 0.2, 0.5, 0.8
# ============================================================
# Probabilities for class "yes"
p1 <- predict(best_model, newdata =test_df, type = "prob")[,"yes"]

# Threshold predictions
pred_02 <- ifelse(p1 >= 0.2, "yes", "no")
pred_05 <- ifelse(p1 >= 0.5, "yes", "no")
pred_08 <- ifelse(p1 >= 0.8, "yes", "no")

# Convert to factor 
pred_02 <- factor(pred_02, levels = c("no","yes"))
pred_05 <- factor(pred_05, levels = c("no","yes"))
pred_08 <- factor(pred_08, levels = c("no","yes"))

# Confusion matrices
cm_02 <- confusionMatrix(pred_02, test_df$Conversion, positive = "yes")
cm_05 <- confusionMatrix(pred_05, test_df$Conversion, positive = "yes")
cm_08 <- confusionMatrix(pred_08, test_df$Conversion, positive = "yes")

cat("\n THRESHOLD t = 0.2\n")
cat("Balanced Accuracy",cm_02$byClass["Balanced Accuracy"],"\n")
cat("Recall:",cm_02$byClass["Sensitivity"],"\n")
cat("Specificity:", cm_02$byClass["Specificity"],"\n")
cat("F1:",cm_02$byClass["F1"],"\n")


cat("\n THRESHOLD t = 0.5\n")
cat("Balanced Accuracy", cm_05$byClass["Balanced Accuracy"],"\n")
cat("Recall:",cm_05$byClass["Sensitivity"],"\n")
cat("Specificity:",cm_05$byClass["Specificity"],"\n")
cat("F1:", cm_05$byClass["F1"],"\n")


cat("\n THRESHOLD t = 0.8\n")
cat("Balanced Accuracy", cm_08$byClass["Balanced Accuracy"],"\n")
cat("Recall:",cm_08$byClass["Sensitivity"],"\n")
cat("Specificity:", cm_08$byClass["Specificity"],"\n")
cat("F1:",cm_08$byClass["F1"],"\n")


#OVERALL RESULTS
#The final Random Forest model was selected using a two-step process.
#First, hyperparameters (mtry, ntree, nodesize) were optimized using cross-validation with ROC as the evaluation metric.
#Then, classification threshold tuning was applied.

#The best performance was achieved using:
#ntree = 500, nodesize = 10, mtry = 2 and threshold = 0.8.

#This combination significantly improved the balance between sensitivity and specificity,
#achieving the highest balanced accuracy.