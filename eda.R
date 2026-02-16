#---------------------------------------
# Exploratory Data Analysis
#---------------------------------------

# -----------------------------
# Load libraries
# -----------------------------
library(dplyr)
library(ggplot2)
library(scales)

# -----------------------------
# Load utils (reuse your functions!)
# -----------------------------
source("utils/utils_preprocessing.R")
source("01_preprocessing.R")
# -----------------------------
# Load dataset
# -----------------------------
marketing_df<-read.csv("marketing.csv")

prep<-preprocess_data(
  data=marketing_df,
  target_col="Conversion",
  id_cols="CustomerID",
  filter_col="ConversionRate",
  filter_min=0,
  filter_max=1,
  train_prop=0.8,
  stratify=TRUE,
  seed=123,
  model_type='glmnet'
)
eda_data<-prep$full_processed_data

#fix the type of the variable conversion

eda_data$Conversion<-as.factor(eda_data$Conversion)

#-------------------------------------------
# Conversion Distribution
#---------------------------------------------
counts_df <- count(eda_data, Conversion)
counts_df$percent <- counts_df$n / sum(counts_df$n)

ggplot(counts_df, aes(x = Conversion, y = percent, fill = Conversion)) +
  geom_col(width = 0.6, show.legend = FALSE) +
  scale_y_continuous(labels = percent_format(), limits = c(0, 1)) +
  scale_fill_manual(values = c("orange", "blue")) +
  labs(
    title = "Distribution of Customer Conversion",
    x = "Conversion",
    y = "Percentage of Customers"
  ) 

#----------------------------
# Conversion vs Gender
#----------------------------

ggplot(eda_data, aes(x = Gender, fill =Conversion)) +
  geom_bar(position = "fill") +
  labs(title = "Conversion Rate by Gender", y = "Proportion") 

#--------------------------------------  
# Campaign Channel vs Conversion
#---------------------------------------

ggplot(eda_data, aes(x = CampaignChannel, fill =Conversion)) +
  geom_bar(position = "fill") +
  scale_y_continuous(labels = percent_format()) +
  labs(title = "Conversion Rate by Campaign Channel", y = "Proportion") 

#---------------------------------------------
# Engagement behaviour Time on site vs Conversion
#---------------------------------------------

ggplot(eda_data, aes(x =Conversion, y = TimeOnSite)) +
  geom_boxplot(fill = "grey") +
  labs(title = "Time on Site vs Conversion") 

ggplot(eda_data, aes(x =Conversion, y = PagesPerVisit)) +
  geom_boxplot(fill = "pink") +
  labs(title = "Pages per Visit vs Conversion") 

#-------------------
#Income & AdSpend
#------------------
ggplot(eda_data, aes(x =Conversion, y = Income)) +
  geom_boxplot(fill = "yellow") +
  labs(title = "Income vs Conversion")


ggplot(eda_data, aes(x = Conversion, y = AdSpend)) +
  geom_boxplot(fill = "grey") +
  labs(title = "Ad Spend vs Conversion") 


