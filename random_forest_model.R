
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
  weights    = obs_weight,
  probability=TRUE
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
  number = 5,
  search="random",  
   classProbs = TRUE,
  summaryFunction = twoClassSummary
)
n_features <- ncol(X_train)

tuneGrid<-expand.grid(
  mtry=seq(2,n_features,by=4),
  splitrule="gini",
  min.node.size=c(1,3,5,8,10)
)

set.seed(123)

rf_tuned<-train(
  Conversion~.,
  data=train_df,
  method="ranger",
  trControl=control,
  tuneGrid=tuneGrid,
  metric="ROC",
  num.trees=500,
  weights=obs_weight,
  importance="impurity"
  
)

cat("\n BEST TUNED PARAMETERS \n")
print(rf_tuned$bestTune)

tuned_probabilities<-predict(rf_tuned,newdata=test_df, type="prob")[,"yes"]
tuned_predictions<-factor(ifelse(tuned_probabilities>0.5,"yes","no"),levels=c("no","yes"))




#EVALUATION OF TUNED MODEL

tuned_confmatrix<-confusionMatrix(data=tuned_predictions,test_df$Conversion,positive="yes")
roc_tuned<-roc(test_df$Conversion,tuned_probabilities,quiet=TRUE)
auc_roc_tuned<-as.numeric(auc(roc_tuned))


cat("\n  TUNED RF RESULTS:")
cat("Balanced Accuracy:",round(tuned_confmatrix$byClass["Balanced Accuracy"],4),"\n")
cat("Recall:",round(tuned_confmatrix$byClass["Sensitivity"],4),"\n")
cat("Specificity :",round(tuned_confmatrix$byClass["Specificity"],4),"\n")
cat("F1:",round(tuned_confmatrix$byClass["F1"],4),"\n")
cat("AUC-ROC:",round(auc_roc_tuned,4),"\n")



#VISUALISATION

#Importance of Features
#The plot shows the importance of each feature on predicting the target variable(conversion)
importance_matrix <-rf_tuned$finalModel$variable.importance #importance(best_model$finalModel)

imp_df <- data.frame(
  Feature =names(importance_matrix),
  Importance = importance_matrix
)
imp_df <- imp_df %>%
  arrange(desc(Importance))
ggplot(imp_df[1:15,], aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_col(fill = "grey") +
  coord_flip()+
  labs(
    title = "Top 15 Features in Random Forest",
    x = "",
    y = "Mean Decrease Gini"
  ) +
  theme_minimal()
png("featureImportace.png")
dev.off()


#ROC CURVE PLOT
png("roc_curve_tuning.png")
roc_curve <- roc(test_df$Conversion, tuned_probabilities)
plot(roc_curve,
     col = "blue",
     main = "ROC Curve - Tuned Random Forest")
dev.off()

#ROC-AUC CURVE PLOT
png("auc_roc_curve2.png")
auc_value <- auc(roc_curve)
plot(roc_curve,
     col = "blue",
     lwd = 2,
     legacy.axes = TRUE,
     main = paste("ROC Curve (AUC =", round(auc_value, 3), ")"))

abline(a = 0, b = 1, lty = 2, col = "grey")
dev.off()
