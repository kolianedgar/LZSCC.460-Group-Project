#===========================
#RANDOM FOREST MODEL
#===========================

library(randomForest)
library(caret)
library(dplyr)
library(pROC)
library(PRROC)
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
obs_weight <- prep_rf$obs_weights

# Create dataframe for modeling
train_df <- data.frame(X_train, Conversion = y_train)
test_df  <- data.frame(X_test,  Conversion = y_test)

# Convert to proper factor levels for caret
train_df$Conversion <- factor(train_df$Conversion, levels = c("0","1"), labels = c("no","yes"))
test_df$Conversion  <- factor(test_df$Conversion,  levels = c("0","1"), labels = c("no","yes"))

# ============================================================
# BASELINE RANDOM FOREST 
# ============================================================
cat("\n============================================================\n")
cat("TRAINING BASELINE RANDOM FOREST\n")
cat("============================================================\n\n")

set.seed(123)
rf_baseline <- randomForest(
  Conversion ~ .,
  data = train_df,
  mtry = floor(sqrt(ncol(X_train))),
  nodesize = 1,
  ntree = 200,
  weights = obs_weight,
  importance = TRUE
)

# Baseline evaluation
prob_baseline <- predict(rf_baseline, newdata = test_df, type = "prob")[,"yes"]
predicted_baseline <- factor(ifelse(prob_baseline > 0.5, "yes", "no"), levels = c("no","yes"))

baseline_confmatrix <- confusionMatrix(
  data      = predicted_baseline,
  reference = test_df$Conversion,
  positive  = "yes"
)

roc_baseline <- roc(test_df$Conversion, prob_baseline, quiet = TRUE)
auc_roc_baseline <- as.numeric(auc(roc_baseline))

cat("\n=== BASELINE RF RESULTS ===\n")
cat("Balanced Accuracy:", round(baseline_confmatrix$byClass["Balanced Accuracy"], 4), "\n")
cat("Accuracy:         ", round(baseline_confmatrix$overall["Accuracy"], 4), "\n")
cat("Recall:           ", round(baseline_confmatrix$byClass["Sensitivity"], 4), "\n")
cat("Specificity:      ", round(baseline_confmatrix$byClass["Specificity"], 4), "\n")
cat("F1:               ", round(baseline_confmatrix$byClass["F1"], 4), "\n")
cat("AUC-ROC:          ", round(auc_roc_baseline, 4), "\n\n")

# ============================================================
# HYPERPARAMETER TUNING WITH CROSS-VALIDATION
# Hyperparameter Tuning(mtry, nodesize, ntree) 
# ============================================================


n_features <- ncol(X_train)

#defining mtry hyperparameter
mtry_values <- c(
  floor(sqrt(n_features)),           # default
  floor(n_features / 2)              # 1/2 of features
)

#defining nodesize hyperparameter
nodesize_values <- c( 5, 10)
ntree_values <- c(200, 500)

#hyperparameter grid
param_grid <- expand.grid(
  mtry = mtry_values,
  nodesize = nodesize_values,
  ntree = ntree_values
)


# 5-fold CV manually
set.seed(123)
n_folds <- 5 #number of folds (5-cross validation)
folds <- createFolds(train_df$Conversion, k = n_folds, list = TRUE)


best_auc<-0
best_mtry<-NULL
best_nodesize<-NULL
best_ntree<-NULL

#grid search
for (i in 1:nrow(param_grid)){
  params<-param_grid[i,]
  fold_aucs<-c() #empty vector of number of folds

  #cross validation
  for(fold in 1:n_folds){
    
    #split data
    val_idx<-folds[[fold]]
    train_cv<-train_df[-val_idx,]
    val_cv<-train_df[val_idx,]
    
    weights_cv<-obs_weight[-val_idx]
    
    #train model with current hyperparameters
    rf_cv<- randomForest(
      Conversion~.,
      data=train_cv,
      mtry=params$mtry,
      nodesize=params$nodesize,
      ntree=params$ntree,
      weights=weights_cv
    )
    
    #predictions on validation fold
    
    probability_cv<-predict(rf_cv,newdata=val_cv, type="prob")[,"yes"]
    
    #AUC
    roc_cv<-roc(val_cv$Conversion, probability_cv,quiet=TRUE)
    auc_value<-as.numeric(auc(roc_cv))
    
    #adds the auc to the vector
    fold_aucs<-c(fold_aucs,auc_value)
  }
  
  current_auc<-mean(fold_aucs)
  
  if(current_auc>best_auc){
    best_auc<-current_auc
    best_mtry<-params$mtry
    best_nodesize<-params$nodesize
    best_ntree<-params$ntree
  }
    
}


#BEST HYPERPARAMETERS

cat("mtry:     ", best_mtry, "\n")
cat("nodesize: ", best_nodesize, "\n")
cat("ntree:    ", best_ntree, "\n")
cat("CV AUC:   ", round(best_auc, 4),"\n")



#training final model with the best hyperparameters
set.seed(123)

rf_tuned <- randomForest(
  Conversion ~ .,
  data = train_df,
  mtry = best_mtry,
  nodesize = best_nodesize,
  ntree = best_ntree,
  weights = obs_weight,
  importance = TRUE
)

#train and evaluation on test set

tuned_probabilities<-predict(rf_tuned,newdata=test_df,type="prob")[,"yes"]
tuned_predictions<-factor(ifelse(tuned_probabilities>0.5,"yes","no"),levels=c("no","yes"))

#confusion matrix
tuned_confmatrix<-confusionMatrix(
  data=tuned_predictions,
  reference=test_df$Conversion,
  positive="yes"
)


#ROC-AUC
roc_tuned<-roc(test_df$Conversion, tuned_probabilities,quiet=TRUE)
auc_roc_tuned<-as.numeric(auc(roc_tuned))

#PR-AUC
labels<-ifelse(test_df$Conversion=="yes",1,0)
pr_obj<-pr.curve(scores.class0 = tuned_probabilities, weights.class0 =labels, curve=TRUE)
auc_pr<-pr_obj$auc.integral

# Extract metrics
precision <- as.numeric(tuned_confmatrix$byClass["Precision"])
recall <- as.numeric(tuned_confmatrix$byClass["Sensitivity"])
specificity <- as.numeric(tuned_confmatrix$byClass["Specificity"])
f1 <- as.numeric(tuned_confmatrix$byClass["F1"])
accuracy <- as.numeric(tuned_confmatrix$overall["Accuracy"])
balanced_accuracy <- as.numeric(tuned_confmatrix$byClass["Balanced Accuracy"])

cat("=== TUNED RF RESULTS ===\n")
cat("Balanced Accuracy:", round(balanced_accuracy, 4), "\n")
cat("Accuracy:         ", round(accuracy, 4), "\n")
cat("Recall:           ", round(recall, 4), "\n")
cat("Specificity:      ", round(specificity, 4), "\n")
cat("Precision:        ", round(precision, 4), "\n")
cat("F1:               ", round(f1, 4), "\n")
cat("AUC-ROC:          ", round(auc_roc_tuned, 4), "\n")
cat("AUC-PR:           ", round(auc_pr, 4), "\n\n")

# save final results
results_rf <- data.frame(
  Model = "Random Forest (Tuned)",
  mtry = best_mtry,
  nodesize = best_nodesize,
  ntree = best_ntree,
  AUC_ROC = auc_roc_tuned,
  PR_AUC = auc_pr,
  Accuracy = accuracy,
  Balanced_Accuracy = balanced_accuracy,
  Sensitivity = recall,
  Specificity = specificity,
  Precision = precision,
  F1 = f1
)

write.csv(results_rf, "rf_final_results.csv", row.names = FALSE)

# display the results table
View(results_rf)

#VISUALISATION

#Importance of Features
#The plot shows the importance of each feature on predicting the target variable(conversion)
importance_matrix <-importance(rf_tuned)#rf_tuned$finalModel$variable.importance #importance(best_model$finalModel)

imp_df <- data.frame(
  Feature =rownames(importance_matrix),
  Importance = importance_matrix[,"MeanDecreaseAccuracy"]
)

#orders the features from most important to less
imp_df <- imp_df %>%
  arrange(desc('Importance'))

png("featureImportance3.png")
ggplot(imp_df[1:15,], aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_col(fill = "grey") +
  coord_flip()+
  labs(
    title = "Top 15 Features in Random Forest",
    x = "",
    y = "Mean Decrease Accuracy"
  ) +
  theme_minimal()
dev.off()


#ROC-AUC CURVE PLOT
png("auc_roc_curve2.png")
auc_value <- auc(roc_curve)
plot(roc_curve,
     col = "blue",
     lwd = 2,
     legacy.axes = TRUE,
     main = paste("ROC Curve (AUC =", round(auc_value,3)))

abline(a = 0, b = 1, lty = 2, col = "grey")
dev.off()


#PR CURVE

png("pr_auc_curve_rf.png")
plot(pr_obj,
     col="blue",
     lwd=2,
     main=paste("PR CURVE (Precision Recall Curve (PRAUC = ",round(pr_obj$auc.integral,4),")")
)

abline(h=sum(labels)/length(labels),
       lty=2,
       col="blue",
       lwd=2
       )
dev.off()


