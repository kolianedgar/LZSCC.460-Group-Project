library(dplyr)
library(ggplot2)
library(glmnet)
library(caret)
library(ggcorrplot)


source("utils/utils.R")
source("01_preprocessing.R")

set_script_wd()


# 2. Data Loading and Preprocessing

marketing_data <- read.csv("marketing.csv")

prep_result <- preprocess_data(
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

df_eda <- prep_result$full_processed_data

# Create a folder for plots if it does not exist
if (!dir.exists("plots")) {
  dir.create("plots")
}

# Plot 1: Target Variable Distribution (Conversion)
# Check for class imbalance
p1 <- ggplot(df_eda, aes(x = as.factor(Conversion), fill = as.factor(Conversion))) +
  geom_bar() +
  labs(title = "Distribution of Conversion", x = "Conversion (0=No, 1=Yes)") +
  theme_minimal()

print(p1)
ggsave("plots/01_target_distribution.png", plot = p1)

# Plot 2: Age Distribution
# Check the age range of customers
p2 <- ggplot(df_eda, aes(x = Age)) +
  geom_histogram(binwidth = 5, fill = "skyblue", color = "black") +
  labs(title = "Customer Age Distribution", x = "Age", y = "Count") +
  theme_minimal()

print(p2)
ggsave("plots/02_age_distribution.png", plot = p2)


# Plot 3: Conversion Rate by Campaign Channel
# Identify which channel performs best
p3 <- ggplot(df_eda, aes(x = CampaignChannel, fill = as.factor(Conversion))) +
  geom_bar(position = "fill") +
  scale_y_continuous(labels = scales::percent) +
  labs(title = "Conversion Rate by Channel", y = "Proportion", fill = "Conversion") +
  theme_minimal()

print(p3)
ggsave("plots/03_conversion_by_channel.png", plot = p3)


# Plot 4: Income vs Conversion
# Do higher income customers convert more?
p4 <- ggplot(df_eda, aes(x = as.factor(Conversion), y = Income, fill = as.factor(Conversion))) +
  geom_boxplot() +
  labs(title = "Income Distribution by Conversion", x = "Conversion", y = "Income") +
  theme_minimal()

print(p4)
ggsave("plots/04_income_vs_conversion.png", plot = p4)

# Plot 5: Time On Site vs Conversion
# Does time spent on the website affect conversion?
p5 <- ggplot(df_eda, aes(x = as.factor(Conversion), y = TimeOnSite, fill = as.factor(Conversion))) +
  geom_boxplot() +
  labs(title = "Time On Site by Conversion", x = "Conversion", y = "Minutes on Site") +
  theme_minimal()

print(p5)
ggsave("plots/05_time_vs_conversion.png", plot = p5)


# Select only numeric columns for correlation
numeric_cols <- df_eda %>%
  select(where(is.numeric)) %>%
  select(-matches("CustomerID")) # Remove ID if it exists

# Calculate correlation matrix
cor_matrix <- cor(numeric_cols, use = "complete.obs")

# Plot 6: Correlation Heatmap
# Check for multicollinearity (relationships between variables)
p6 <- ggcorrplot(cor_matrix,
                 method = "square",
                 type = "lower",
                 lab = TRUE,
                 lab_size = 3,
                 colors = c("blue", "white", "red"),
                 title = "Correlation Matrix of Numeric Features")

print(p6)
ggsave("plots/06_correlation_heatmap.png", plot = p6)


# Plot 7: Campaign Channel x Campaign Type Heatmap
# To identify the "Winning Combination" of strategy.
# Calculating Conversion Rate for each combination
heatmap_data <- df_eda %>%
  group_by(CampaignChannel, CampaignType) %>%
  summarise(
    ConversionRate = mean(as.numeric(as.character(Conversion))),
    Count = n(),
    .groups = 'drop'
  )

p7 <- ggplot(heatmap_data, aes(x = CampaignChannel, y = CampaignType, fill = ConversionRate)) +
  geom_tile(color = "white") +
  geom_text(aes(label = scales::percent(ConversionRate, accuracy = 0.1)), color = "white", fontface = "bold") +
  scale_fill_gradient(low = "#3498db", high = "#e74c3c") + # Blue to Red
  labs(title = "Conversion Rate Heatmap: Channel vs Type",
       subtitle = "Red indicates high performance combinations",
       x = "Channel", y = "Campaign Type", fill = "Conv. Rate") +
  theme_minimal()

print(p7)
ggsave("plots/07_strategy_heatmap.png", plot = p7)


# Plot 8: Engagement Scatter Plot (Time on Site vs Pages per Visit)
# To see if Browsers (high engagement) are actually Buyers.
p8 <- ggplot(df_eda, aes(x = TimeOnSite, y = PagesPerVisit, color = as.factor(Conversion))) +
  geom_point(alpha = 0.5, size = 2) +
  labs(title = "Engagement Analysis: Time vs Pages",
       subtitle = "Do engaged users convert more?",
       x = "Time on Site (min)", y = "Pages per Visit", color = "Converted") +
  scale_color_manual(values = c("0" = "gray", "1" = "red")) +
  theme_minimal() +
  theme(legend.position = "top")

print(p8)
ggsave("plots/08_engagement_scatter.png", plot = p8)


# Plot 9: Loyalty vs Income Density
# To understand the profile of High-Value Customers.
p9 <- ggplot(df_eda, aes(x = Income, y = LoyaltyPoints)) +
  geom_bin2d(bins = 30) +
  scale_fill_viridis_c() +
  facet_wrap(~Conversion) + # Split by Conversion status
  labs(title = "Customer Profile: Income vs Loyalty Points",
       subtitle = "Left: Non-Converted, Right: Converted",
       x = "Annual Income", y = "Loyalty Points", fill = "Count") +
  theme_minimal()

print(p9)
ggsave("plots/09_loyalty_income_density.png", plot = p9)


# Plot 10: AdSpend Distribution by Conversion (Overall)
# To check if converted users generally cost more to acquire.
p10 <- ggplot(df_eda, aes(x = as.factor(Conversion), y = AdSpend, fill = as.factor(Conversion))) +
  geom_boxplot() +
  labs(title = "AdSpend Distribution: Converted vs Non-Converted",
       subtitle = "Do we spend more on users who convert?",
       x = "Conversion (0=No, 1=Yes)", y = "Ad Spend (€)") +
  scale_fill_manual(values = c("0" = "gray70", "1" = "#2ecc71")) +
  theme_minimal() +
  theme(legend.position = "none")

print(p10)
ggsave("plots/10_adspend_overall.png", plot = p10)


# Plot 11: AdSpend Impact by Channel (Faceted Boxplot)
# To identify WHICH channel requires high spend to convert.
p11 <- ggplot(df_eda, aes(x = as.factor(Conversion), y = AdSpend, fill = as.factor(Conversion))) +
  geom_boxplot() +
  facet_wrap(~CampaignChannel) +
  labs(title = "AdSpend Efficiency by Channel",
       subtitle = "Where does spending more money actually lead to conversion?",
       x = "Conversion", y = "Ad Spend (€)") +
  scale_fill_manual(values = c("0" = "gray70", "1" = "#3498db")) +
  theme_minimal() +
  theme(legend.position = "none")

print(p11)
ggsave("plots/11_adspend_by_channel.png", plot = p11)


# Plot 12: AdSpend ROI Analysis (Binning)
# To see if spending TOO much has diminishing returns.
adspend_bins <- df_eda %>%
  mutate(SpendGroup = cut_number(AdSpend, n = 5, labels = c("Very Low", "Low", "Medium", "High", "Very High"))) %>%
  group_by(SpendGroup) %>%
  summarise(
    ConversionRate = mean(as.numeric(as.character(Conversion))),
    AvgSpend = mean(AdSpend),
    .groups = 'drop'
  )

p12 <- ggplot(adspend_bins, aes(x = AvgSpend, y = ConversionRate)) +
  geom_line(group = 1, color = "darkblue", size = 1) +
  geom_point(size = 3, color = "red") +
  geom_text(aes(label = scales::percent(ConversionRate, accuracy = 0.1)), vjust = -0.5) +
  labs(title = "AdSpend ROI Curve",
       subtitle = "Does Conversion Rate increase linearly with Spend?",
       x = "Average Ad Spend (€)", y = "Conversion Rate") +
  theme_minimal()

print(p12)
ggsave("plots/12_adspend_roi_curve.png", plot = p12)