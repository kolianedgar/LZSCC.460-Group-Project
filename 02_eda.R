# ============================================================
# 02_eda.R — Exploratory Data Analysis for Marketing Conversion
# ============================================================

#-----------------------------------
# Set working directory
#-----------------------------------
setwd("C:/Users/sejdi/Desktop/Lancaster University Masters - Leipzig/Data Science Fundamentals/LZSCC.460-Group-Project")

#-----------------------------------
# Load libraries and utilities
#-----------------------------------
library(ggplot2)
library(dplyr)
library(tidyr)
library(corrplot)
library(GGally)
library(scales)

source("utils/utils.R")
source("01_preprocessing.R")

#-----------------------------------
# Load and preprocess data
#-----------------------------------
marketing_data <- read.csv("marketing.csv")

prep <- preprocess_data(
  data       = marketing_data,
  target_col = "Conversion",
  id_cols    = "CustomerID",
  filter_col = "ConversionRate",
  filter_min = 0,
  filter_max = 1,
  train_prop = 0.8,
  stratify   = TRUE,
  seed       = 123,
  model_type = "glmnet"
)

df <- prep$full_processed_data

# Ensure Conversion is a factor with readable labels
df$Conversion <- factor(df$Conversion, levels = c(0, 1), labels = c("No", "Yes"))

# Verify both classes exist
cat("Conversion counts:\n")
print(table(df$Conversion))

#-----------------------------------
# Output directory for plots
#-----------------------------------
plot_dir <- "plots/"
if (!dir.exists(plot_dir)) dir.create(plot_dir, recursive = TRUE)

# ============================================================
# 1. TARGET VARIABLE — Class Balance
# ============================================================

p1 <- ggplot(df, aes(x = Conversion, fill = Conversion)) +
  geom_bar() +
  geom_text(stat = "count", aes(label = after_stat(count)), vjust = -0.5) +
  scale_fill_manual(values = c("No" = "#E74C3C", "Yes" = "#2ECC71")) +
  labs(title = "Distribution of Conversion", x = "Conversion", y = "Count") +
  theme_minimal() +
  theme(legend.position = "none")

ggsave(file.path(plot_dir, "01_conversion_balance.png"), p1, width = 6, height = 5)

# ============================================================
# 2. DEMOGRAPHICS vs Conversion
# ============================================================

# 2a. Age distribution by Conversion
p2a <- ggplot(df, aes(x = Age, fill = Conversion)) +
  geom_density(alpha = 0.5) +
  scale_fill_manual(values = c("No" = "#E74C3C", "Yes" = "#2ECC71")) +
  labs(title = "Age Distribution by Conversion", x = "Age", y = "Density") +
  theme_minimal()

ggsave(file.path(plot_dir, "02a_age_density.png"), p2a, width = 7, height = 5)

# 2b. Income distribution by Conversion
p2b <- ggplot(df, aes(x = Income, fill = Conversion)) +
  geom_density(alpha = 0.5) +
  scale_fill_manual(values = c("No" = "#E74C3C", "Yes" = "#2ECC71")) +
  labs(title = "Income Distribution by Conversion", x = "Income (EUR)", y = "Density") +
  theme_minimal()

ggsave(file.path(plot_dir, "02b_income_density.png"), p2b, width = 7, height = 5)

# 2c. Conversion counts by Gender
p2c <- ggplot(df, aes(x = Gender, fill = Conversion)) +
  geom_bar(position = "dodge") +
  geom_text(stat = "count", aes(label = after_stat(count)),
            position = position_dodge(width = 0.9), vjust = -0.5) +
  scale_fill_manual(values = c("No" = "#E74C3C", "Yes" = "#2ECC71")) +
  labs(title = "Conversion by Gender", x = "Gender", y = "Count") +
  theme_minimal()

ggsave(file.path(plot_dir, "02c_gender_conversion.png"), p2c, width = 6, height = 5)

# ============================================================
# 3. CAMPAIGN CHARACTERISTICS vs Conversion
# ============================================================

# 3a. Conversion counts by Campaign Channel
p3a <- ggplot(df, aes(x = CampaignChannel, fill = Conversion)) +
  geom_bar(position = "dodge") +
  geom_text(stat = "count", aes(label = after_stat(count)),
            position = position_dodge(width = 0.9), vjust = -0.5, size = 3) +
  scale_fill_manual(values = c("No" = "#E74C3C", "Yes" = "#2ECC71")) +
  labs(title = "Conversion by Campaign Channel", x = "Campaign Channel", y = "Count") +
  theme_minimal()

ggsave(file.path(plot_dir, "03a_channel_conversion.png"), p3a, width = 7, height = 5)

# 3b. Conversion counts by Campaign Type
p3b <- ggplot(df, aes(x = CampaignType, fill = Conversion)) +
  geom_bar(position = "dodge") +
  geom_text(stat = "count", aes(label = after_stat(count)),
            position = position_dodge(width = 0.9), vjust = -0.5, size = 3) +
  scale_fill_manual(values = c("No" = "#E74C3C", "Yes" = "#2ECC71")) +
  labs(title = "Conversion by Campaign Type", x = "Campaign Type", y = "Count") +
  theme_minimal()

ggsave(file.path(plot_dir, "03b_type_conversion.png"), p3b, width = 7, height = 5)

# 3c. AdSpend by Conversion
p3c <- ggplot(df, aes(x = Conversion, y = AdSpend, fill = Conversion)) +
  geom_boxplot(outlier.alpha = 0.3) +
  scale_fill_manual(values = c("No" = "#E74C3C", "Yes" = "#2ECC71")) +
  labs(title = "Ad Spend by Conversion", x = "Conversion", y = "Ad Spend (EUR)") +
  theme_minimal() +
  theme(legend.position = "none")

ggsave(file.path(plot_dir, "03c_adspend_boxplot.png"), p3c, width = 6, height = 5)

# ============================================================
# 4. ENGAGEMENT METRICS vs Conversion (individual boxplots)
# ============================================================

# 4a. ClickThroughRate
p4a <- ggplot(df, aes(x = Conversion, y = ClickThroughRate, fill = Conversion)) +
  geom_boxplot(outlier.alpha = 0.3) +
  scale_fill_manual(values = c("No" = "#E74C3C", "Yes" = "#2ECC71")) +
  labs(title = "Click-Through Rate by Conversion", x = "Conversion", y = "Click-Through Rate") +
  theme_minimal() + theme(legend.position = "none")

ggsave(file.path(plot_dir, "04a_ctr_boxplot.png"), p4a, width = 6, height = 5)

# 4b. ConversionRate
p4b <- ggplot(df, aes(x = Conversion, y = ConversionRate, fill = Conversion)) +
  geom_boxplot(outlier.alpha = 0.3) +
  scale_fill_manual(values = c("No" = "#E74C3C", "Yes" = "#2ECC71")) +
  labs(title = "Conversion Rate (Feature) by Conversion", x = "Conversion", y = "Conversion Rate") +
  theme_minimal() + theme(legend.position = "none")

ggsave(file.path(plot_dir, "04b_convrate_boxplot.png"), p4b, width = 6, height = 5)

# 4c. WebsiteVisits
p4c <- ggplot(df, aes(x = Conversion, y = WebsiteVisits, fill = Conversion)) +
  geom_boxplot(outlier.alpha = 0.3) +
  scale_fill_manual(values = c("No" = "#E74C3C", "Yes" = "#2ECC71")) +
  labs(title = "Website Visits by Conversion", x = "Conversion", y = "Website Visits") +
  theme_minimal() + theme(legend.position = "none")

ggsave(file.path(plot_dir, "04c_visits_boxplot.png"), p4c, width = 6, height = 5)

# 4d. PagesPerVisit
p4d <- ggplot(df, aes(x = Conversion, y = PagesPerVisit, fill = Conversion)) +
  geom_boxplot(outlier.alpha = 0.3) +
  scale_fill_manual(values = c("No" = "#E74C3C", "Yes" = "#2ECC71")) +
  labs(title = "Pages Per Visit by Conversion", x = "Conversion", y = "Pages Per Visit") +
  theme_minimal() + theme(legend.position = "none")

ggsave(file.path(plot_dir, "04d_pages_boxplot.png"), p4d, width = 6, height = 5)

# 4e. TimeOnSite
p4e <- ggplot(df, aes(x = Conversion, y = TimeOnSite, fill = Conversion)) +
  geom_boxplot(outlier.alpha = 0.3) +
  scale_fill_manual(values = c("No" = "#E74C3C", "Yes" = "#2ECC71")) +
  labs(title = "Time on Site by Conversion", x = "Conversion", y = "Time on Site (min)") +
  theme_minimal() + theme(legend.position = "none")

ggsave(file.path(plot_dir, "04e_timeonsite_boxplot.png"), p4e, width = 6, height = 5)

# 4f. SocialShares
p4f <- ggplot(df, aes(x = Conversion, y = SocialShares, fill = Conversion)) +
  geom_boxplot(outlier.alpha = 0.3) +
  scale_fill_manual(values = c("No" = "#E74C3C", "Yes" = "#2ECC71")) +
  labs(title = "Social Shares by Conversion", x = "Conversion", y = "Social Shares") +
  theme_minimal() + theme(legend.position = "none")

ggsave(file.path(plot_dir, "04f_socialshares_boxplot.png"), p4f, width = 6, height = 5)

# 4g. EmailOpens
p4g <- ggplot(df, aes(x = Conversion, y = EmailOpens, fill = Conversion)) +
  geom_boxplot(outlier.alpha = 0.3) +
  scale_fill_manual(values = c("No" = "#E74C3C", "Yes" = "#2ECC71")) +
  labs(title = "Email Opens by Conversion", x = "Conversion", y = "Email Opens") +
  theme_minimal() + theme(legend.position = "none")

ggsave(file.path(plot_dir, "04g_emailopens_boxplot.png"), p4g, width = 6, height = 5)

# 4h. EmailClicks
p4h <- ggplot(df, aes(x = Conversion, y = EmailClicks, fill = Conversion)) +
  geom_boxplot(outlier.alpha = 0.3) +
  scale_fill_manual(values = c("No" = "#E74C3C", "Yes" = "#2ECC71")) +
  labs(title = "Email Clicks by Conversion", x = "Conversion", y = "Email Clicks") +
  theme_minimal() + theme(legend.position = "none")

ggsave(file.path(plot_dir, "04h_emailclicks_boxplot.png"), p4h, width = 6, height = 5)

# ============================================================
# 5. HISTORICAL BEHAVIOR vs Conversion
# ============================================================

# 5a. PreviousPurchases by Conversion
p5a <- ggplot(df, aes(x = Conversion, y = PreviousPurchases, fill = Conversion)) +
  geom_boxplot(outlier.alpha = 0.3) +
  scale_fill_manual(values = c("No" = "#E74C3C", "Yes" = "#2ECC71")) +
  labs(title = "Previous Purchases by Conversion", x = "Conversion", y = "Previous Purchases") +
  theme_minimal() + theme(legend.position = "none")

ggsave(file.path(plot_dir, "05a_prev_purchases_boxplot.png"), p5a, width = 6, height = 5)

# 5b. LoyaltyPoints by Conversion
p5b <- ggplot(df, aes(x = Conversion, y = LoyaltyPoints, fill = Conversion)) +
  geom_boxplot(outlier.alpha = 0.3) +
  scale_fill_manual(values = c("No" = "#E74C3C", "Yes" = "#2ECC71")) +
  labs(title = "Loyalty Points by Conversion", x = "Conversion", y = "Loyalty Points") +
  theme_minimal() + theme(legend.position = "none")

ggsave(file.path(plot_dir, "05b_loyalty_boxplot.png"), p5b, width = 6, height = 5)

# ============================================================
# 6. CORRELATION HEATMAP
# ============================================================

num_df <- df %>% select(where(is.numeric))
cor_matrix <- cor(num_df, use = "complete.obs")

png(file.path(plot_dir, "06_correlation_heatmap.png"), width = 900, height = 800)
corrplot(
  cor_matrix,
  method      = "color",
  type        = "lower",
  tl.col      = "black",
  tl.cex      = 0.8,
  addCoef.col = "black",
  number.cex  = 0.6,
  title       = "Correlation Matrix of Numeric Variables",
  mar         = c(0, 0, 2, 0)
)
dev.off()

# ============================================================
# 7. INTERACTION: AdSpend vs Conversion by Campaign Channel
# ============================================================

p7 <- ggplot(df, aes(x = Conversion, y = AdSpend, fill = Conversion)) +
  geom_boxplot(outlier.alpha = 0.3) +
  scale_fill_manual(values = c("No" = "#E74C3C", "Yes" = "#2ECC71")) +
  facet_wrap(~ CampaignChannel, scales = "free_y") +
  labs(title = "Ad Spend by Conversion across Campaign Channels",
       x = "Conversion", y = "Ad Spend (EUR)") +
  theme_minimal() + theme(legend.position = "bottom")

ggsave(file.path(plot_dir, "07_adspend_by_channel.png"), p7, width = 12, height = 6)

# ============================================================
# 8. SCATTERPLOT: ClickThroughRate vs ConversionRate
# ============================================================

p8 <- ggplot(df, aes(x = ClickThroughRate, y = ConversionRate, color = Conversion)) +
  geom_point(alpha = 0.3, size = 1) +
  scale_color_manual(values = c("No" = "#E74C3C", "Yes" = "#2ECC71")) +
  labs(title = "Click-Through Rate vs Conversion Rate",
       x = "Click-Through Rate", y = "Conversion Rate") +
  theme_minimal()

ggsave(file.path(plot_dir, "08_ctr_vs_convrate.png"), p8, width = 7, height = 5)

# ============================================================
# 9. PAIR PLOT: Key Engagement Variables
# ============================================================

pair_cols <- c("ClickThroughRate", "WebsiteVisits", "PagesPerVisit",
               "TimeOnSite", "EmailClicks", "Conversion")

p9 <- ggpairs(
  df[, pair_cols],
  mapping  = aes(color = Conversion, alpha = 0.4),
  upper    = list(continuous = wrap("cor", size = 3)),
  lower    = list(continuous = wrap("points", size = 0.5)),
  diag     = list(continuous = wrap("densityDiag")),
  progress = FALSE
) +
  scale_color_manual(values = c("No" = "#E74C3C", "Yes" = "#2ECC71")) +
  scale_fill_manual(values = c("No" = "#E74C3C", "Yes" = "#2ECC71")) +
  theme_minimal()

ggsave(file.path(plot_dir, "09_pair_plot.png"), p9, width = 14, height = 12)

# ============================================================
# 10. OUTLIER INSPECTION (raw data before capping)
# ============================================================

raw_clean <- drop_duplicates(marketing_data) %>% select(-CustomerID)
raw_imputer <- fit_imputer(raw_clean %>% select(-Conversion))
raw_clean_features <- apply_imputer(raw_clean %>% select(-Conversion), raw_imputer)

outlier_vars <- c("Income", "AdSpend", "WebsiteVisits", "SocialShares",
                  "EmailOpens", "EmailClicks", "LoyaltyPoints")

raw_long <- raw_clean_features %>%
  select(all_of(outlier_vars)) %>%
  pivot_longer(cols = everything(), names_to = "Variable", values_to = "Value")

p10 <- ggplot(raw_long, aes(x = Variable, y = Value)) +
  geom_boxplot(fill = "#3498DB", outlier.alpha = 0.3) +
  facet_wrap(~ Variable, scales = "free", ncol = 4) +
  labs(title = "Outlier Inspection — Key Numeric Variables", x = "", y = "Value") +
  theme_minimal() + theme(axis.text.x = element_blank())

ggsave(file.path(plot_dir, "10_outlier_inspection.png"), p10, width = 14, height = 8)

# ============================================================
# 11. MISSING VALUE SUMMARY (raw data)
# ============================================================

raw_missing <- colSums(is.na(marketing_data))
raw_missing <- raw_missing[raw_missing > 0]

if (length(raw_missing) > 0) {
  missing_df <- data.frame(
    Variable = names(raw_missing),
    Count    = as.integer(raw_missing)
  )
  
  p11 <- ggplot(missing_df, aes(x = reorder(Variable, Count), y = Count)) +
    geom_col(fill = "#3498DB") +
    geom_text(aes(label = Count), hjust = -0.2) +
    coord_flip() +
    labs(title = "Missing Values by Variable (Raw Data)", x = "Variable", y = "Missing Count") +
    theme_minimal()
  
  ggsave(file.path(plot_dir, "11_missing_values.png"), p11, width = 7, height = 5)
} else {
  cat("No missing values found in raw data.\n")
}

# ============================================================
# 12. SUMMARY STATISTICS TABLE
# ============================================================

summary_stats <- df %>%
  group_by(Conversion) %>%
  summarise(
    across(where(is.numeric), list(
      mean   = ~ mean(.x, na.rm = TRUE),
      median = ~ median(.x, na.rm = TRUE),
      sd     = ~ sd(.x, na.rm = TRUE)
    )),
    .groups = "drop"
  )

write.csv(summary_stats, file.path(plot_dir, "12_summary_stats.csv"), row.names = FALSE)

cat("\n========================================\n")
cat("EDA complete. All plots saved to:", plot_dir, "\n")
cat("========================================\n")