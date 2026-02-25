# Objective 3: Logistic Regression for factor interpretation

library(dplyr)
library(ggplot2)
library(broom)
library(caret)
library(pROC)

source("utils/utils.R")
source("01_preprocessing.R")
set_script_wd()

df <- read.csv("marketing.csv")

# prep data using existing pipeline (glmnet handles dummy vars and scaling)
prep <- preprocess_data(
  data = df,
  target_col = "Conversion",
  id_cols = "CustomerID",
  filter_col = "ConversionRate",
  filter_min = 0,
  filter_max = 1,
  train_prop = 0.8,
  stratify = TRUE,
  seed = 123,
  model_type = "glmnet"
)

# setup training set
train_df <- as.data.frame(prep$X_train)
train_df$Conversion <- prep$y_train

# fit model (quasibinomial prevents warnings with non-integer obs_weights)
mod <- glm(
  Conversion ~ .,
  data = train_df,
  family = quasibinomial(link = "logit"),
  weights = prep$obs_weights
)

# evaluation on test set
test_df <- as.data.frame(prep$X_test)
y_test <- prep$y_test

probs <- predict(mod, newdata = test_df, type = "response")
preds <- factor(ifelse(probs > 0.5, "1", "0"), levels = levels(y_test))

cm <- confusionMatrix(preds, y_test, positive = "1")
roc_obj <- roc(y_test, probs, quiet = TRUE)

cat("\nTest Set Metrics\n")
cat(sprintf("Accuracy:          %.4f\n", cm$overall["Accuracy"]))
cat(sprintf("Sensitivity:       %.4f\n", cm$byClass["Sensitivity"]))
cat(sprintf("Specificity:       %.4f\n", cm$byClass["Specificity"]))
cat(sprintf("Balanced Accuracy: %.4f\n", cm$byClass["Balanced Accuracy"]))
cat(sprintf("AUC:               %.4f\n", auc(roc_obj)))

if (!dir.exists("plots")) dir.create("plots")

png("plots/14_roc_curve.png", width = 800, height = 800, res = 150)
plot(roc_obj, col = "steelblue", lwd = 2, main = "ROC Curve: Logistic Regression")
abline(a = 1, b = -1, lty = 2, col = "darkgray")
dev.off()

# extracting insights
results <- tidy(mod, conf.int = TRUE) %>%
  filter(term != "(Intercept)") %>%
  mutate(
    OR = exp(estimate),
    ci_low = exp(conf.low),
    ci_high = exp(conf.high),
    is_sig = ifelse(p.value < 0.05, "Yes", "No")
  ) %>%
  arrange(p.value)

cat("\nTop Drivers\n")
print(head(results %>% select(term, OR, p.value, is_sig), 10))

# forest plot
p <- ggplot(results, aes(x = reorder(term, OR), y = OR, color = is_sig)) +
  geom_point(size = 2.5) +
  geom_errorbar(aes(ymin = ci_low, ymax = ci_high), width = 0.2) +
  geom_hline(yintercept = 1, linetype = "dashed", color = "firebrick", alpha = 0.7) +
  coord_flip() +
  scale_color_manual(values = c("Yes" = "seagreen", "No" = "gray70")) +
  labs(
    title = "Logistic Regression: Impact on Conversion Odds",
    x = "",
    y = "Odds Ratio (95% CI)"
  ) +
  theme_minimal() +
  theme(legend.position = "none", panel.grid.minor = element_blank())

ggsave("plots/13_statistical_drivers_forest.png", plot = p, width = 8, height = 7, bg = "white")