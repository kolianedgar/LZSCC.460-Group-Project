#-----------------------------------
# Load necessary libraries and files
#-----------------------------------

library(dplyr)
library(ggplot2)
library(tidyr)

source("utils_preprocessing.R")

#----------------
# Read data
#----------------

marketing_data <- read.csv(file.path(getwd(), "marketing.csv"), header = TRUE, sep = ",")

#------------------------
# Add a row number column
#------------------------

marketing_data <- marketing_data %>%
  mutate(row_index = row_number())

#---------------------------------
# Display rows with missing values
#---------------------------------

marketing_data %>%
  filter(if_any(everything(), ~ is.na(.) | . == "NA"))

#--------------------------------------------------------
# Perform conditional median imputation on missing values
# Condition: Gender of the customer
#--------------------------------------------------------

marketing_data <- df_imputed <- impute_by_group_median(
  data = marketing_data,
  group_col = "Gender"
)

#---------------------------------------------------
# Drop unnecessary columns
# Columns produced as a result of running imputation
#---------------------------------------------------

marketing_data <- marketing_data %>% 
  select(-Income_missing, -row_index,
         -Age_missing, -CampaignChannel_missing,
         -CampaignType_missing, -ClickThroughRate_missing,
         -Conversion_missing, -ConversionRate_missing,
         -CustomerID_missing, -EmailClicks_missing,
         -EmailOpens_missing, -LoyaltyPoints_missing,
         -PagesPerVisit_missing, -PreviousPurchases_missing,
         -row_index_missing, -SocialShares_missing, -TimeOnSite_missing,
         -WebsiteVisits_missing, -AdSpend_missing)

#-----------------------------------------------
# Prepare and plot boxplots of numeric variables
# Identify existing outliers
#-----------------------------------------------

df_long <- marketing_data %>%
  select(where(is.numeric)) %>%
  pivot_longer(
    cols = everything(),
    names_to = "variable",
    values_to = "value"
  )

ggplot(df_long, aes(x = variable, y = value)) +
  geom_boxplot(outlier.alpha = 0.6) +
  facet_wrap(~ variable, scales = "free_y") +
  labs(
    title = "Boxplots of Numeric Variables",
    x = NULL,
    y = NULL
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_blank(),
    axis.ticks.x = element_blank()
  )

#-------------------------------------------
# Get rid of outliers from numerical columns
#-------------------------------------------

marketing_data_clean <- df_clean <- remove_outliers_iqr(marketing_data)

#-----------------------------------------------
# Prepare and plot boxplots of numeric variables
# Show that there are no outliers
#-----------------------------------------------

df_long_clean <- marketing_data_clean %>%
  select(where(is.numeric)) %>%
  pivot_longer(
    cols = everything(),
    names_to = "variable",
    values_to = "value"
  )

ggplot(df_long_clean, aes(x = variable, y = value)) +
  geom_boxplot(outlier.alpha = 0.6) +
  facet_wrap(~ variable, scales = "free_y") +
  labs(
    title = "Boxplots of Numeric Variables",
    x = NULL,
    y = NULL
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_blank(),
    axis.ticks.x = element_blank()
  )
