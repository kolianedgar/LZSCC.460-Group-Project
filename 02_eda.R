# 02_eda.R — Exploratory Data Analysis for Marketing Conversion

# This file contains all the code for exploring and visualising the marketing dataset
# The goal is to understand patterns in the data before building any models
# We look at things like class balance, demographics, campaign channels, and ad spend
# All plots are saved to the plots/ folder


# Setting the working directory
setwd("C:/Users/sejdi/Desktop/Lancaster University Masters - Leipzig/Data Science Fundamentals/LZSCC.460-Group-Project")


# Loading the libraries we need for this analysis
# Each library serves a specific purpose in building our plots and handling data

library(ggplot2)    # The main plotting library - used for all our charts
library(dplyr)      # For data manipulation like filtering, grouping, and summarising
library(tidyr)      # For reshaping data - turning wide tables into long format
library(scales)     # For formatting numbers nicely on plots (e.g. percentages)
library(patchwork)  # For combining multiple plots side by side into one image
library(gridExtra)  # For arranging tables and plots together in a grid layout
library(grid)       # For adding custom text elements like titles above tables

# Loading our utility functions and preprocessing pipeline
# utils.R contains helper functions like set_script_wd()
# 01_preprocessing.R contains the preprocess_data() function that cleans our data
source("utils/utils.R")
source("01_preprocessing.R")


# Loading the raw dataset from the CSV file
# This file contains 8200 rows and 18 columns of marketing campaign data
marketing_data <- read.csv("marketing.csv")

# Running our preprocessing pipeline on the raw data
# This handles duplicate removal, missing value imputation, outlier capping, etc.
# We use model_type = "glmnet" here because the EDA uses the full_processed_data
# which is created before encoding and scaling, so the model type doesn't matter
# for the EDA itself - it only affects how the train/test sets are prepared
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

# We use the full processed dataset for EDA (not the train/test split)
# This gives us the cleaned version of all 8198 observations
# It has been imputed and outlier-capped but NOT encoded or scaled
# so the original variable names and values are still readable
df <- prep$full_processed_data

# Converting the Conversion column from 0/1 numbers to "No"/"Yes" labels
# This makes our plots more readable since the legend will say "No" and "Yes"
# instead of just 0 and 1
df$Conversion <- factor(df$Conversion, levels = c(0, 1), labels = c("No", "Yes"))

# Creating a numeric version of Conversion as well
# We need this for calculating rates and percentages later
# For example, mean(Conversion_numeric) gives us the conversion rate
df$Conversion_numeric <- ifelse(df$Conversion == "Yes", 1, 0)

# Quick check to make sure both classes exist in our data
# If one class was accidentally removed during preprocessing, this would catch it
cat("Conversion counts:\n")
print(table(df$Conversion))

# Setting up the output folder where all plots will be saved
# If the folder doesn't exist yet, we create it
plot_dir <- "plots/"
if (!dir.exists(plot_dir)) dir.create(plot_dir, recursive = TRUE)


# 1. Target Variable & Demographics vs Conversion
# This creates a combined plot with four panels side by side:
# Panel 1: Bar chart showing how many customers converted vs didn't (class balance)
# Panel 2: Age density curves overlaid for converters and non-converters
# Panel 3: Income density curves overlaid for converters and non-converters
# Panel 4: Gender bar chart split by conversion status

# Panel 1 - Class Balance bar chart
# Shows 988 non-converters vs 7010 converters
# The geom_text line adds the exact count number above each bar
p1 <- ggplot(df, aes(x = Conversion, fill = Conversion)) +
  geom_bar() +
  geom_text(stat = "count", aes(label = after_stat(count)), vjust = -0.5) +
  scale_fill_manual(values = c("No" = "#E74C3C", "Yes" = "#2ECC71")) +
  labs(title = "Class Balance", x = "Conversion", y = "Count") +
  theme_minimal() + theme(legend.position = "none")

# Panel 2 - Age Distribution
# Density plot shows the shape of the age distribution for each group
# If the two curves overlap heavily, age doesn't help separate the groups
# alpha = 0.5 makes them semi-transparent so we can see both curves
p2a <- ggplot(df, aes(x = Age, fill = Conversion)) +
  geom_density(alpha = 0.5) +
  scale_fill_manual(values = c("No" = "#E74C3C", "Yes" = "#2ECC71")) +
  labs(title = "Age Distribution", x = "Age", y = "Density") +
  theme_minimal() + theme(legend.position = "none")

# Panel 3 - Income Distribution
# Same idea as age - we want to see if income separates the two groups
p2b <- ggplot(df, aes(x = Income, fill = Conversion)) +
  geom_density(alpha = 0.5) +
  scale_fill_manual(values = c("No" = "#E74C3C", "Yes" = "#2ECC71")) +
  labs(title = "Income Distribution", x = "Income (EUR)", y = "Density") +
  theme_minimal() + theme(legend.position = "none")

# Panel 4 - Conversion by Gender
# Dodged bar chart showing male vs female counts for each conversion status
# position = "dodge" puts the bars side by side instead of stacked
p2c <- ggplot(df, aes(x = Gender, fill = Conversion)) +
  geom_bar(position = "dodge") +
  geom_text(stat = "count", aes(label = after_stat(count)),
            position = position_dodge(width = 0.9), vjust = -0.5) +
  scale_fill_manual(values = c("No" = "#E74C3C", "Yes" = "#2ECC71")) +
  labs(title = "Conversion by Gender", x = "Gender", y = "Count") +
  theme_minimal()

# Combining all four panels into one wide image using patchwork
# The | operator places plots side by side
# plot_annotation adds a title above the entire combined plot
p_combined <- (p1 | p2a | p2b | p2c) +
  plot_annotation(title = "Target Distribution & Demographics vs Conversion")

# Saving the combined plot as a PNG file
# width = 18 makes it wide enough to fit all four panels without squishing
ggsave(file.path(plot_dir, "01_target_and_demographics.png"), p_combined, width = 18, height = 5)


# 2. Campaign Channel Distribution
# This plot shows how many customers came through each campaign channel
# We want to check if the data is balanced across channels or if one dominates

# First we make CampaignChannel a factor so ggplot treats it as categories
df$CampaignChannel <- factor(df$CampaignChannel)

# We calculate the count and percentage for each channel
# The Label column combines both into a readable string like "1718 (21.5%)"
channel_summary <- df %>%
  group_by(CampaignChannel) %>%
  summarise(Count = n(), .groups = 'drop') %>%
  mutate(Percentage = Count / sum(Count) * 100,
         Label = paste0(Count, " (", round(Percentage, 1), "%)")) %>%
  arrange(Count)

# We reorder the factor levels by count so the bars appear sorted in the plot
channel_summary$CampaignChannel <- factor(channel_summary$CampaignChannel,
                                          levels = channel_summary$CampaignChannel)

# Building the horizontal bar chart
# coord_flip() turns the vertical bars horizontal for easier reading
# hjust = -0.1 places the label text just outside the end of each bar
p_channel_dist <- ggplot(channel_summary, aes(x = CampaignChannel, y = Count, fill = CampaignChannel)) +
  geom_bar(stat = "identity", width = 0.7, color = "black", size = 0.5) +
  geom_text(aes(label = Label), hjust = -0.1, size = 4, fontface = "bold") +
  coord_flip() +
  scale_fill_manual(values = c("PPC" = "#3498DB", "Referral" = "#E74C3C",
                               "SEO" = "#2ECC71", "Social Media" = "#9B59B6",
                               "Email" = "#F39C12")) +
  labs(title = "Distribution of Campaign Channels",
       subtitle = paste0("Total Observations: ", sum(channel_summary$Count)),
       x = "Campaign Channel", y = "Number of Observations") +
  scale_y_continuous(expand = expansion(mult = c(0, 0.15)),
                     limits = c(0, max(channel_summary$Count) * 1.15)) +
  theme_minimal(base_size = 14) +
  theme(plot.title = element_text(face = "bold", size = 16, hjust = 0.5),
        plot.subtitle = element_text(size = 12, hjust = 0.5, color = "gray40"),
        legend.position = "none",
        panel.grid.major.y = element_blank())

# Saving at high resolution (dpi = 300) for clear presentation
ggsave(file.path(plot_dir, "02_campaign_channel_distribution.png"), p_channel_dist,
       width = 10, height = 6, dpi = 300, bg = "white")


# 4a. Conversion by Campaign Channel
# This plot shows how many customers converted vs didn't within each channel
# Unlike plot 2 which just showed total counts, this one splits by conversion status
# This helps us see if some channels produce more conversions than others

p4a <- ggplot(df, aes(x = CampaignChannel, fill = Conversion)) +
  geom_bar(position = "dodge") +
  geom_text(stat = "count", aes(label = after_stat(count)),
            position = position_dodge(width = 0.9), vjust = -0.5, size = 3) +
  scale_fill_manual(values = c("No" = "#E74C3C", "Yes" = "#2ECC71")) +
  labs(title = "Conversion by Campaign Channel", x = "Campaign Channel", y = "Count") +
  theme_minimal()

ggsave(file.path(plot_dir, "04a_channel_conversion.png"), p4a, width = 7, height = 5)


# 7. Conversion Rate by Campaign Type
# This plot calculates the percentage of customers who converted for each campaign type
# For example, if 100 customers were in a "Conversion" campaign and 93 converted,
# the conversion rate would be 93%

# First we calculate the conversion rate for each campaign type
# mean(Conversion_numeric) gives us the proportion, then we multiply by 100 for percentage
type_conversion <- df %>%
  group_by(CampaignType) %>%
  summarise(
    Total = n(),
    Conversions = sum(Conversion_numeric),
    ConvRate = mean(Conversion_numeric) * 100,
    .groups = 'drop'
  ) %>%
  arrange(desc(ConvRate))

# Reorder the factor levels so the highest conversion rate appears first
type_conversion$CampaignType <- factor(type_conversion$CampaignType,
                                       levels = type_conversion$CampaignType)

# Building the bar chart with percentage labels above each bar
p7 <- ggplot(type_conversion, aes(x = CampaignType, y = ConvRate, fill = CampaignType)) +
  geom_bar(stat = "identity", width = 0.7, color = "black", size = 0.5) +
  geom_text(aes(label = paste0(round(ConvRate, 1), "%")),
            vjust = -0.5, size = 5, fontface = "bold") +
  scale_fill_manual(values = c("Awareness" = "#3498DB", "Consideration" = "#2ECC71",
                               "Conversion" = "#F39C12", "Retention" = "#9B59B6")) +
  labs(title = "Conversion Rate by Campaign Type",
       subtitle = "Which funnel stages perform best?",
       x = "Campaign Type", y = "Conversion Rate (%)") +
  scale_y_continuous(expand = expansion(mult = c(0, 0.15)),
                     limits = c(0, max(type_conversion$ConvRate) * 1.12)) +
  theme_minimal(base_size = 14) +
  theme(plot.title = element_text(face = "bold", size = 16, hjust = 0.5),
        plot.subtitle = element_text(size = 12, hjust = 0.5, color = "gray40"),
        legend.position = "none",
        panel.grid.major.x = element_blank())

ggsave(file.path(plot_dir, "07_conversion_by_type.png"), p7,
       width = 10, height = 6, dpi = 300, bg = "white")


# 8. Conversion Rate by Channel x Type
# This is the most detailed campaign plot - it shows conversion rates
# for every combination of channel and campaign type
# For example: "PPC + Conversion campaign" or "Email + Awareness campaign"
# This helps identify which specific combinations work best

# We group by both channel and type and calculate conversion rate for each combo
channel_type_conversion <- df %>%
  group_by(CampaignChannel, CampaignType) %>%
  summarise(
    Total = n(),
    Conversions = sum(Conversion_numeric),
    ConvRate = mean(Conversion_numeric) * 100,
    .groups = 'drop'
  )

# Building a grouped bar chart where each channel has 4 bars (one per type)
# position = "dodge" places the 4 campaign type bars side by side within each channel
# The angle = 45 on x-axis text prevents channel names from overlapping
p8 <- ggplot(channel_type_conversion, aes(x = CampaignChannel, y = ConvRate, fill = CampaignType)) +
  geom_bar(stat = "identity", position = "dodge", color = "black", size = 0.5) +
  geom_text(aes(label = paste0(round(ConvRate, 1), "%")),
            position = position_dodge(width = 0.9),
            vjust = -0.5, size = 3) +
  scale_fill_manual(values = c("Awareness" = "#3498DB", "Consideration" = "#2ECC71",
                               "Conversion" = "#F39C12", "Retention" = "#9B59B6")) +
  labs(title = "Conversion Rate by Campaign Channel x Campaign Type",
       subtitle = "Best performing channel-type combinations",
       x = "Campaign Channel", y = "Conversion Rate (%)",
       fill = "Campaign Type") +
  scale_y_continuous(expand = expansion(mult = c(0, 0.15))) +
  theme_minimal(base_size = 14) +
  theme(plot.title = element_text(face = "bold", size = 16, hjust = 0.5),
        plot.subtitle = element_text(size = 11, hjust = 0.5, color = "gray40"),
        axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = "right",
        panel.grid.major.x = element_blank())

ggsave(file.path(plot_dir, "08_conversion_channel_type.png"), p8,
       width = 12, height = 6, dpi = 300, bg = "white")


# 11. AdSpend ROI Curve
# This plot shows the relationship between how much was spent on ads
# and what conversion rate was achieved
# We split ad spend into 5 equal-sized groups and calculate the average
# conversion rate for each group, then connect them with a line

# cut_number splits the data into 5 groups with roughly equal numbers of customers
# We then calculate the average conversion rate and average spend for each group
adspend_bins <- df %>%
  mutate(SpendGroup = cut_number(AdSpend, n = 5,
                                 labels = c("Very Low", "Low", "Medium", "High", "Very High"))) %>%
  group_by(SpendGroup) %>%
  summarise(
    ConvRate = mean(Conversion_numeric),
    AvgSpend = mean(AdSpend),
    .groups = 'drop'
  )

# Building a line chart with points at each group
# The red dots mark each spend group and the labels show the exact conversion rate
# percent() from the scales library formats the numbers as percentages
p11 <- ggplot(adspend_bins, aes(x = AvgSpend, y = ConvRate)) +
  geom_line(group = 1, color = "darkblue", size = 1) +
  geom_point(size = 3, color = "red") +
  geom_text(aes(label = percent(ConvRate, accuracy = 0.1)), vjust = -0.5) +
  labs(title = "AdSpend ROI Curve",
       subtitle = "Does Conversion Rate increase linearly with Spend?",
       x = "Average Ad Spend (EUR)", y = "Conversion Rate") +
  theme_minimal()

ggsave(file.path(plot_dir, "11_adspend_roi_curve.png"), p11, width = 8, height = 5)


# 13. Summary Statistics (table image) — Horizontal layout
# This creates a side-by-side table image showing descriptive statistics
# for all numeric variables, split by conversion status
# Left table = Non-Converted customers, Right table = Converted customers
# Each table shows Mean, SD, Median, Min, and Max for every numeric variable

# First we reshape the data from wide to long format
# This means instead of one row per customer with many columns,
# we get many rows per customer with one column for the variable name
# and one column for the value - this makes it easy to group and summarise
summary_long <- df %>%
  select(where(is.numeric), Conversion, -Conversion_numeric) %>%
  pivot_longer(cols = -Conversion, names_to = "Variable", values_to = "Value") %>%
  group_by(Variable, Conversion) %>%
  summarise(
    Mean   = round(mean(Value, na.rm = TRUE), 2),
    SD     = round(sd(Value, na.rm = TRUE), 2),
    Median = round(median(Value, na.rm = TRUE), 2),
    Min    = round(min(Value, na.rm = TRUE), 2),
    Max    = round(max(Value, na.rm = TRUE), 2),
    .groups = "drop"
  )

# Splitting the summary into two separate tables
# One for non-converters and one for converters
table_no  <- summary_long %>% filter(Conversion == "No") %>% select(-Conversion)
table_yes <- summary_long %>% filter(Conversion == "Yes") %>% select(-Conversion)

# Creating the PNG image with both tables side by side
# We use grid.arrange to place the titles and tables in a 2x2 grid
# Row 1: two title labels ("Non-Converted" and "Converted")
# Row 2: the two data tables
# heights = c(1, 12) means the title row is much shorter than the table row
png(file.path(plot_dir, "13_summary_table.png"), width = 2200, height = 1000, res = 150)
grid.arrange(
  grid::textGrob("Non-Converted (No)",
                 gp = grid::gpar(fontsize = 12, fontface = "bold")),
  grid::textGrob("Converted (Yes)",
                 gp = grid::gpar(fontsize = 12, fontface = "bold")),
  tableGrob(table_no, rows = NULL, theme = ttheme_minimal(
    core    = list(fg_params = list(fontsize = 9), padding = unit(c(6, 6), "mm")),
    colhead = list(fg_params = list(fontsize = 10, fontface = "bold"), padding = unit(c(6, 6), "mm"))
  )),
  tableGrob(table_yes, rows = NULL, theme = ttheme_minimal(
    core    = list(fg_params = list(fontsize = 9), padding = unit(c(6, 6), "mm")),
    colhead = list(fg_params = list(fontsize = 10, fontface = "bold"), padding = unit(c(6, 6), "mm"))
  )),
  ncol = 2, nrow = 2,
  heights = c(1, 12)
)
dev.off()

cat("\nEDA complete. All plots saved to:", plot_dir, "\n")