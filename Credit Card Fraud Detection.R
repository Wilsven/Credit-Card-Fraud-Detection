# Require & Load Libraries
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(readr)) install.packages("readr",  repos = "http://cran.us.r-project.org")
if(!require(knitr)) install.packages("knitr", repos = "http://cran.us.r-project.org")
if(!require(kableExtra)) install.packages("kableExtra", repos = "http://cran.us.r-project.org")
if(!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")
if(!require(scales)) install.packages("scales",  repos = "http://cran.us.r-project.org")
if(!require(gridExtra)) install.packages("gridExtra", repos = "http://cran.us.r-project.org")
if(!require(geosphere)) install.packages("geosphere", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(rpart)) install.packages("rpart",  repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("randomForest",  repos = "http://cran.us.r-project.org")
if(!require(tinytex)) install.packages("tinytex",  repos = "http://cran.us.r-project.org")
if(!require(rmarkdown)) install.packages("rmarkdown",  repos = "http://cran.us.r-project.org")

library(tidyverse)
library(data.table)
library(readr)
library(knitr)
library(kableExtra)
library(gridExtra) 
library(scales)
library(lubridate)
library(geosphere)
library(caret)
library(rpart)
library(randomForest)
library(tinytex)
library(rmarkdown)

options(scipen = 999)

#####   Importing Credit Card Fraud data sets   #####

# Credit card fraud dataset obtained from Kaggle at: 
# https://www.kaggle.com/kartik2112/fraud-detection

# Create temp file and download zip file containing dataset
dl <- tempfile()
download.file("https://github.com/Wilsven/Credit-Card-Fraud-Detection/releases/download/v1-files/fraud_data.zip", dl)

# Unzip
unzip(dl)

# Read & import csv files
fraudTest <- read_csv(unzip(dl,"fraudTest.csv"))
fraudTrain <- read_csv(unzip(dl,"fraudTrain.csv"))

# Remove tempfile
rm(dl)

#####   Exploring Pre-split FraudTrain Data   #####

# Exploring data set variables & structure
glimpse(fraudTrain)

# Proportion of fraudulent transactions
prop.table(table(fraudTrain$is_fraud))

# Summary of data
summary(fraudTrain)

# Date ranges for provided train and test sets 
range(fraudTrain$trans_date_trans_time)
range(fraudTest$trans_date_trans_time)

#####   Resampling training & test sets   #####

# Data was pre-split by date. Merging together to randomly split. Must first remove X1 row number to prevent duplicates.
fraudTest <- fraudTest[-1]
fraudTrain <- fraudTrain[-1]
fraudSet <- rbind(fraudTrain, fraudTest)

# removing pre-split sets
rm(fraudTest, fraudTrain)

# Check for NA values in is_fraud column, if present, drop the row(s)
sum(is.na(fraudSet$is_fraud))
index <- which(is.na(fraudSet$is_fraud))
fraudSet <- fraudSet[-index,]

## Splitting data set options
# Creating Random Sampling of Training & 20% Final Test set of fraud data. 
set.seed(1, sample.kind="Rounding") 
test_index <- createDataPartition(y = fraudSet$is_fraud, times = 1, p = 0.2, list = FALSE)
train_set <- fraudSet[-test_index,]
test_set <- fraudSet[test_index,]

# Proportion of fraudulent transactions
prop.table(table(train_set$is_fraud)) #.52%
prop.table(table(test_set$is_fraud)) #.52%

# Removing  
rm(fraudSet)
rm(index)

#####   Exploring Data   #####

### Fraud ###

# Fraud vs legitimate average transaction amounts & number of transactions
train_set %>% 
  group_by(is_fraud) %>% 
  summarize(avg_trans = mean(amt), 
            med_trans = median(amt),
            amt = sum(amt),
            n = n()) %>%
  mutate(pct_amt = (amt/sum(amt))*100,
         pct_n = (n/sum(n))*100)

# Summary of Bins by Fraud
train_set %>%
  mutate(bins = cut(amt, breaks = c(-Inf, 100, 1000, Inf), 
                      labels = c("<$100", "$100-$999", ">$1K+"))) %>%
  group_by(bins, is_fraud) %>%
  summarize(amt = sum(amt),
            n = n()) %>%
  mutate(pct_amt = (amt/sum(amt))*100,
         pct_n = (n/sum(n))*100)

# Plot histogram distributions of legitimate transaction amounts
train_set %>%
  filter(is_fraud == 0) %>%
  mutate(bins = cut(amt, breaks = c(-Inf, 100, 1000, Inf), 
                    labels = c("<$100", "$100-$999", ">$1K+"))) %>%
  ggplot(aes(amt)) +
  geom_histogram(bins = 40, fill = "#56B4E9") +
  facet_wrap(~ bins, scales = "free") +
  labs(title = "Legitimate Transaction Amounts",
       x = "Amount ($)", y = "No. of Transactions") +
  theme_bw(base_size = 10) +
  theme(plot.title = element_text(size = 10))

# Plot histogram distributions of fraudulent transaction amounts
train_set %>%
  filter(is_fraud == 1) %>%
  mutate(bins = cut(amt, breaks = c(-Inf, 100, 1000, Inf), 
                    labels = c("<$100", "$100-$999", ">$1K+"))) %>%
  ggplot(aes(amt)) +
  geom_histogram(bins = 40, fill = "#56B4E9") +
  facet_wrap(~ bins, scales = "free") +
  labs(title = "Fraudulent Transaction Amounts",
       x = "Amount ($)", y = "No. of Transactions") +
  theme_bw(base_size = 10) +
  theme(plot.title = element_text(size = 10))

##### Categories #####

# Legitimate Transaction Amounts by Category
x <- train_set %>%
  filter(is_fraud == 0) %>%
  ggplot(aes(amt)) +
  geom_histogram(bins = 40, fill = "#56B4E9") +
  facet_wrap(~ category) +
  labs(title = "Legitimate Transaction Amounts by Category",
       x = "Amount ($)", y = "No. of Transactions") +
  theme_bw(base_size = 10) +
  theme(plot.title = element_text(size = 10), 
        axis.text.x = element_text(angle = 90))

# Fraudulent Transaction Amounts by Category
y <- train_set %>%
  filter(is_fraud == 1) %>%
  ggplot(aes(amt)) +
  geom_histogram(bins = 40, fill = "#56B4E9") +
  facet_wrap(~ category) +
  labs(title = "Fraudulent Transaction Amounts by Category",
       x = "Amount ($)", y = "No. of Transactions") +
  theme_bw(base_size = 10) +
  theme(plot.title = element_text(size = 10))

# arrange histograms side by side
grid.arrange(x, y, ncol = 2)
# remove variable
rm(x, y)

# Barchart with Fraud & Legit Transactions by Category 
train_set %>%
  group_by(category, is_fraud) %>%
  summarize(amt = sum(amt), n = n()) %>%
  ggplot(aes(x = amt, y = reorder(category, amt), fill = as.factor(is_fraud))) +
  geom_bar(stat = "identity") +
  labs(title = "Transaction Amounts by Category",
       x = "Amount ($)", y = "Category",
       fill = "") +
  scale_fill_manual(values = c("grey68","darkorange2"),
                    labels = c("Legitimate", "Fraudulent")) +
  scale_x_continuous(labels = comma) +  
  theme_bw(base_size = 10) +
  theme(plot.title = element_text(size = 10),
        legend.key.height = unit(.4, "cm"),
        legend.position = "bottom")

# Boxplot with Fraud & Legit Transactions by Category 
train_set %>%
  filter(amt < 1200) %>%
  mutate(is_fraud = as.factor(is_fraud)) %>%
  group_by(is_fraud, category) %>%
  ggplot(aes(x = is_fraud, y = amt, colour = is_fraud)) +
  geom_boxplot(lwd = .5,
               outlier.size = .75,
               outlier.alpha = .3) +
  facet_wrap(~ category, ncol = 4) +
  labs(title = "Transaction Amounts by Category: Legitimate vs Fraudulent", 
       y = "Amount ($)",
       colour = "") +
  scale_y_continuous(breaks = c(0,250,500,750,1000)) +
  scale_colour_manual(values = c("grey68","darkorange2"),
                    labels = c("Legitimate", "Fraudulent")) +
  theme_bw(base_size = 10) +
  theme(legend.position = "bottom", 
        panel.grid.minor = element_blank(), 
        axis.text.x=element_blank(), 
        axis.title.x=element_blank(),  
        plot.title = element_text(size = 10))
  
# (Grid) Legitimate & Fraudulent Breakdown per Category by % Amount 
train_set %>%
  mutate(bins = cut(amt, breaks = c(-Inf, 100, 250, 800, 1400, Inf),
         labels = c("<$100", "$100-$249", "$250-$799", "$800-$1400", "$1.4K+")),
         is_fraud = as.factor(is_fraud)) %>%
  group_by(is_fraud, category, bins) %>%
  summarize(amt = sum(amt)) %>%
  mutate(pct_amt = (amt/sum(amt))) %>%
  ggplot(aes(x = pct_amt, y = reorder(category, pct_amt), fill = is_fraud)) +
  geom_col(position = position_dodge2(width = 0.9, preserve = "single")) +
  facet_wrap(~ bins, ncol = 5) +
  scale_fill_manual(values = c("grey68","darkorange2"),
                      labels = c("Legitimate", "Fraudulent")) +
  labs(title = "Breakdown of Transaction Amounts by Category",
       x = "% of Amount", y = "Category", fill = "") +
  theme_bw(base_size = 10) +
  theme(legend.position = "bottom", 
        panel.grid.minor = element_blank(), 
        axis.text.x = element_text(angle = 90),
        legend.key.height = unit(.4, "cm"),
        plot.title = element_text(size = 10))

##### Timeline of an Account with Fraudulent Transactions #####

# Pick the tenth (randomly selected) credit card with fraudulent transactions
cc_number <- train_set[which(train_set$is_fraud == 1),][10,]$cc_num

# Plot a scatterplot as a timeline to visualize nature of transactions
train_set %>%
  filter(cc_num == cc_number) %>%
  mutate(trans_date = as_date(trans_date_trans_time),
         is_fraud = as.factor(is_fraud)) %>%
  ggplot(aes(x = trans_date, y = amt, colour = is_fraud)) +
  geom_point() + 
  scale_colour_manual(values = c("grey68","darkorange2"),
                    labels = c("Legitimate", "Fraudulent")) +
  labs(title = "Example Timeline of CC with Fraudulent Transactions",
       x = "Date", y = "Amount ($)", colour = "") +
  theme_bw(base_size = 10) +
  theme(legend.position = "bottom", 
        panel.grid.minor = element_blank(),
        plot.title = element_text(size = 10))

# Plot Percentage of Fraudulent Transactions Amounts throughout the Day
train_set %>%
  mutate(trans_hour = hour(trans_date_trans_time)) %>%
  group_by(trans_hour, is_fraud) %>%
  summarize(amt = sum(amt)) %>%
  mutate(pct_amt = (amt/sum(amt))*100) %>%
  filter(is_fraud == 1) %>%
  ggplot(aes(x = trans_hour)) +
  geom_bar(aes(y = pct_amt, fill = pct_amt), stat = "identity") +
  geom_text(aes(y = pct_amt, label = sprintf("%1.1f%%", pct_amt)), size = 2, nudge_y = 1) +
  scale_x_continuous(breaks = seq(0, 23),
                     labels = c("12AM", "1AM", "2AM", "3AM", "4AM", "5AM", "6AM", "7AM", "8AM",
                                "9AM", "10AM", "11AM", "12PM", "1PM", "2PM", "3PM", "4PM", "5PM", 
                                "6PM", "7PM", "8PM", "9PM", "10PM", "11PM")) +
  scale_y_continuous(labels = function(x){
    paste0(x,"%")}, 
    limits = c(0,40)) + 
  scale_fill_continuous(labels = function(x)
    {paste0(x, "%")},
    limits = c(0,40)) +
  labs(title = "% of Fraudulent Transaction Amounts throughout the Day",
       x = "Time", y = "% of Total Fraudulent Transactions", fill = "Percentage") +
  theme_bw(base_size = 10) +
  theme(panel.grid.minor = element_blank(),
        panel.grid.major.x = element_blank(),
        plot.title = element_text(size = 10),
        axis.text.x = element_text(angle = 90))

# Plot Number of Fraudulent Transactions throughout the Day
train_set %>%
  mutate(trans_hour = hour(trans_date_trans_time)) %>%
  group_by(trans_hour, is_fraud) %>%
  summarize(n = n()) %>%
  mutate(pct_n = (n/sum(n))*100) %>%
  ggplot(aes(x = trans_hour, y = pct_n, colour = as.factor(is_fraud))) +
  geom_point() +
  geom_text(aes(y = pct_n, label = sprintf("%1.1f%%", pct_n)), size = 3.5, nudge_y = 1, vjust = -1) +
  scale_x_continuous(breaks = seq(0, 23),
                     labels = c("12AM", "1AM", "2AM", "3AM", "4AM", "5AM", "6AM", "7AM", "8AM",
                                "9AM", "10AM", "11AM", "12PM", "1PM", "2PM", "3PM", "4PM", "5PM", 
                                "6PM", "7PM", "8PM", "9PM", "10PM", "11PM")) +
  scale_y_continuous(labels = function(x){
    paste0(x,"%")}) + 
  scale_colour_manual(values = c("grey68","darkorange2"),
                      labels = c("Legitimate", "Fraudulent")) +
  labs(title = "% of Types of Transactions throughout the Day",
       x = "Time", y = "% of Transactions", colour = "") +
  theme_bw(base_size = 10) +
  theme(panel.grid.minor = element_blank(),
        panel.grid.major.y = element_blank(),
        plot.title = element_text(size = 10),
        axis.text.x = element_text(angle = 90))

# Load lubridate library 
library(lubridate)

# Plot Proportion of Fraudulent Transaction Amounts during each Month
train_set %>%
  mutate(trans_month = month(trans_date_trans_time, label = TRUE)) %>%
  group_by(trans_month, is_fraud) %>%
  summarize(amt = sum(amt)) %>%
  mutate(pct_amt = (amt/sum(amt))*100) %>%
  filter(is_fraud == 1) %>%
  select(-is_fraud, -amt) %>%
  ggplot(aes(x = trans_month)) +
  geom_bar(aes(y = pct_amt, fill = pct_amt), stat = "identity") +
  geom_text(aes(y = pct_amt, label = sprintf("%1.2f%%", pct_amt)), size = 3.5, vjust = -1) +
  scale_y_continuous(labels = function(x){
    paste0(x,"%")}, 
    limits = c(0,8)) + 
  scale_fill_continuous(labels = function(x)
  {paste0(x, "%")},
  limits = c(0,8)) +
  labs(title = "% of Fraudulent Transaction Amounts for each Month",
       x = "Month", y = "% of Fraudulent Transaction Amounts", fill = "Percentage") +
  theme_bw(base_size = 10) +
  theme(panel.grid.minor = element_blank(),
        panel.grid.major.x = element_blank(),
        plot.title = element_text(size = 10))

# Plot Number of Fraudulent Transactions for each Month
train_set %>%
  mutate(trans_month = month(trans_date_trans_time, label = TRUE)) %>%
  group_by(trans_month, is_fraud) %>%
  summarize(n = n()) %>%
  mutate(pct_n = (n/sum(n))*100) %>%
  ggplot(aes(x = trans_month, y = pct_n, colour = as.factor(is_fraud))) +
  geom_point() +
  geom_text(aes(y = pct_n, label = sprintf("%1.2f%%", pct_n)), size = 3.5, nudge_y = 1, vjust = -1) +
  scale_y_continuous(labels = function(x){
    paste0(x,"%")}) + 
  scale_colour_manual(values = c("grey68","darkorange2"),
                      labels = c("Legitimate", "Fraudulent")) +
  labs(title = "% of Types of Transactions for each Month",
       x = "Month", y = "% of Transactions", colour = "") +
  theme_bw(base_size = 10) +
  theme(panel.grid.minor = element_blank(),
        panel.grid.major.y = element_blank(),
        plot.title = element_text(size = 10))

# Plot Proportion of Fraudulent Transaction Amounts during each Day of the Month
train_set %>%
  mutate(trans_day = day(trans_date_trans_time)) %>%
  group_by(trans_day, is_fraud) %>%
  summarize(amt = sum(amt)) %>%
  mutate(pct_amt = (amt/sum(amt))*100) %>%
  filter(is_fraud == 1) %>%
  select(-is_fraud, -amt) %>%
  ggplot(aes(x = trans_day)) +
  geom_bar(aes(y = pct_amt, fill = pct_amt), stat = "identity") +
  geom_text(aes(y = pct_amt, label = sprintf("%1.2f%%", pct_amt)), size = 2.5, vjust = -1) +
  scale_x_continuous(breaks = seq(1,31,1)) +
  scale_y_continuous(labels = function(x){
    paste0(x,"%")}, 
    limits = c(0,6)) + 
  scale_fill_continuous(labels = function(x)
  {paste0(x, "%")},
  limits = c(0,6)) +
  labs(title = "% of Fraudulent Transaction Amounts for each Day of the Month",
       x = "Day", y = "% of Fraudulent Transaction Amounts", fill = "Percentage") +
  theme_bw(base_size = 10) +
  theme(panel.grid.minor = element_blank(),
        panel.grid.major.x = element_blank(),
        plot.title = element_text(size = 10))

# Plot Number of Fraudulent Transactions for each Day of the Month
train_set %>%
  mutate(trans_day = day(trans_date_trans_time)) %>%
  group_by(trans_day, is_fraud) %>%
  summarize(n = n()) %>%
  mutate(pct_n = (n/sum(n))*100) %>%
  ggplot(aes(x = trans_day, y = pct_n, colour = as.factor(is_fraud))) +
  geom_point() +
  geom_text(aes(y = pct_n, label = sprintf("%1.1f%%", pct_n)), size = 3, nudge_y = 1, vjust = -1) +
  scale_x_continuous(breaks = seq(1,31,1)) +
  scale_y_continuous(labels = function(x){
    paste0(x,"%")}) + 
  scale_colour_manual(values = c("grey68","darkorange2"),
                      labels = c("Legitimate", "Fraudulent")) +
  labs(title = "% of Types of Transactions for each Day of the Month",
       x = "Day", y = "% of Transactions", colour = "") +
  theme_bw(base_size = 10) +
  theme(panel.grid.minor = element_blank(),
        panel.grid.major.y = element_blank(),
        plot.title = element_text(size = 10))

# Fraud by gender
train_set %>%
  group_by(is_fraud, gender) %>%
  summarize(amt = sum(amt), n = n()) %>%
  mutate(pct_amt = (amt/sum(amt))*100,
         pct_n = (n/sum(n))*100) %>%
  filter(is_fraud == 1)

# Plot histogram of Fraud by Y.O.B
train_set %>%
  ggplot(aes(dob)) +
  geom_histogram(bins = 40, colour = "white", fill = "#56B4E9") +
  facet_wrap(~ is_fraud, scales = "free") +
  labs(title = "Histogram of Transactions by Y.O.B",
       x = "Year", y = "Count") +
  theme_bw(base_size = 10) +
  theme(plot.title = element_text(size = 10))
  
##### Location Elements #####

### By State ###

# Calculate distance between customer & merchant with longitude and latitude . 
# This takes several minutes. 
train_distance <- train_set[c(1:5,12:14,20:22)] %>% 
  rowwise() %>% 
  mutate(trans_dist = distHaversine(c(long, lat),c(merch_long, merch_lat))/ 1609)

range(train_distance$trans_dist)

# Distribution of distances
train_distance %>% 
  ggplot(aes(trans_dist)) + 
  geom_histogram(bins = 40, colour = "white", fill = "#56B4E9") + 
  facet_wrap(~ is_fraud, scales = "free") +
  labs(title = "Transaction Distances",
       x = "Miles between Customer / Merchant", y = "") +
  theme_bw(base_size = 10) +
  theme(plot.title = element_text(size = 10))
  
rm(train_distance)

# Plot Percentage of Fraudulent Transaction Amounts by State
x <- train_set %>%
  group_by(is_fraud, state) %>%
  summarize(amt = sum(amt), n = n()) %>%
  mutate(pct_amt = (amt/sum(amt))*100,
         pct_n = (n/sum(n))*100) %>%
  filter(is_fraud == 1)  %>%
  ggplot(aes(x = pct_amt, y = reorder(state, pct_amt))) +
  geom_bar(width = 0.8, fill = "#56B4E9", stat = "identity") +
  geom_text(aes(label = sprintf("%1.1f%%", pct_amt)), 
            size = 2.5, nudge_x = 0.25, alpha = 0.5) +
  labs(title = "% of Fraudulent Transaction Amounts by State",
       x = "Percentage", y = "") +
  theme_bw(base_size = 10) +
  theme(axis.title.x=element_blank(),
        panel.grid.minor = element_blank(),
        panel.grid.major.y = element_blank(),
        plot.title = element_text(size = 10))
  
# Plot Breakdown of Accounts by State
y <- train_set %>%
  group_by(state) %>%
  summarize(account = n_distinct(cc_num), amt = sum(amt), n = n()) %>%
  mutate(pct_account = (account/sum(account))*100) %>%
  ggplot(aes(x = pct_account, y = reorder(state, pct_account))) +
  geom_bar(width = 0.8, fill = "#56B4E9", stat = "identity") +
  geom_text(aes(label = sprintf("%1.1f%%", pct_account)), 
            size = 2.5, nudge_x = 0.25, alpha = 0.5) +
  labs(title = "Breakdown of Accounts by State",
       x = "Percentage", y = "") +
  theme_bw(base_size = 10) +
  theme(axis.title.x = element_blank(),
        panel.grid.minor = element_blank(),
        panel.grid.major.y = element_blank(),
        plot.title = element_text(size = 10))

grid.arrange(x, y, ncol = 2)
rm(x, y)

##### Data Preparation #####

### Prepping Final Training Set 

# Creating prepped training set 
# Removing columns not used in modeling
# Removing fraud from merchant
# Added transaction amount bins
train_set <- train_set[c(1:5,22)] 

# Removing columns not used in modeling
# Removing fraud from merchant
# Added transaction amount bins capped at 100000 as if credit limit
train_set <- train_set %>% 
  mutate(merchant = str_remove(merchant, "fraud_"), 
         cc_num = as.character(cc_num), 
         is_fraud = as_factor(is_fraud),
         bins = cut(amt, breaks = c(0,99,249,799,1399,100000),
                    labels = c("<$100","$100-$249","$250-$799","$800-$1400","$1400+")))

# Separating transaction date & time & converting day and trans_hour to factors
# This runs much faster when saved to different object
train_prep <- train_set %>% 
  mutate(month = month(trans_date_trans_time, label = TRUE), 
         day = day(trans_date_trans_time), 
         hour = hour(trans_date_trans_time), 
         weekday = wday(trans_date_trans_time, label = TRUE)) %>% 
  mutate(day = as_factor(day), hour = as_factor(hour))

# Splitting data again at 90/10: for modeling training & testing
# Leaving test_set for final evaluation of models
set.seed(1, sample.kind = "Rounding")
model_index <- createDataPartition(y = train_prep$is_fraud, times = 1, p = 0.1, list = FALSE)
train2 <- train_prep[-model_index,]
test2 <- train_prep[model_index,]

# Proportion of fraudulent transactions
table(train2$is_fraud) # 0.52%
table(test2$is_fraud) # 0.52%

# Removing initial full data set 
rm(train_set, train_prep)

### Prepping Final Validation Test Set 

# Creating prepped validation set 
# Removing columns not used in modeling
# Removing fraud from merchant
# Added transaction amount bins
test_set <- test_set[c(1:5,22)]

# Removing columns not used in modeling
# Removing fraud from merchant
# Added transaction amount bins capped at 100000 as if credit limit
test_prep <- test_set %>%
  mutate(merchant = str_remove(merchant, "fraud_"),
         cc_num = as.character(cc_num),
         is_fraud = as.factor(is_fraud),
         bins = cut(amt, breaks = c(0,99,249,799,1399,100000), 
                    labels = c("<$100","$100-$249","$250-$799","$800-$1400","$1400+")))

# Separating transaction date & time & converting day and trans_hour to factors
# This runs much faster when saved to different object
test_set <- test_prep %>%
  mutate(month = month(trans_date_trans_time, label = TRUE),
         day = day(trans_date_trans_time),
         hour = hour(trans_date_trans_time),
         weekday = wday(trans_date_trans_time, label = TRUE)) %>%
  mutate(day = as.factor(day), hour = as.factor(hour))

# Removing test_prep
rm(test_prep, model_index, test_index)

##### Modeling #####

# Let's calculate loss when there is no Fraud Detection System in place
# Create a vector of 0's of length equal to test set
predictions <- factor(rep(0, times = nrow(test2)), levels = c(0,1))

# Compute cost of not detecting fraudulent transactions from test2
cost <- sum(test2$amt[test2$is_fraud == 1])

# Compute cost of not detecting fraudulent transactions from test_set 
cost2 <- sum(test_set$amt[test_set$is_fraud == 1])

# Compute Accuracy
confusionMatrix(predictions, reference = test2$is_fraud)$overall[["Accuracy"]] # 0.9947838
confusionMatrix(predictions, reference = test2$is_fraud)$byClass[["Balanced Accuracy"]] # 0.5

# Tabulating fraud predictions
cost_preds <- tibble(amt = test2$amt, 
                     results = test2$is_fraud, 
                     no_preds = predictions)

# Tabulating cost results
cost_results <- tibble(Model = "No Fraud Detection",
                       AmtSaved = 0,
                       FraudMissed = cost,
                       Misclassified = 0,
                       SavedPct = 0,
                       MisclassPct = 0,
                       Specificity = 0,
                       NPV = 0)

# Removing large elements
rm(predictions)

# Create a dataframe of prediction possibilities
Actual <- factor(c(0,0,1,1))
Predicted <- factor(c(0, 1, 0, 1))
Y <- c("TP","FN","FP","TN")
cm_df <- data.frame(Actual, Predicted, Y)

# Create a tile plot
cm_df %>%
  ggplot(aes(x = Actual, y = reorder(Predicted, desc(Predicted)), labels = Y)) +
  geom_tile(aes(fill = Y)) +
  geom_text(label = Y, size = 3) +
  scale_fill_brewer() +
  labs(x="Actual",y="Predicted") +  
  theme_bw(base_size = 9) + 
  theme(legend.position = "none")

##### Rpart Model 1 #####

# Fit the training data to the rpart model
# This takes all variables except trans_date_trans_time
fit_rpart_all <- train2 %>% 
  select(-c(trans_date_trans_time)) %>% 
  rpart(is_fraud ~ ., data = ., method = "class")

# Rpart predictions
yhat_rpart_all <- predict(fit_rpart_all, test2, type = "class")

# Rpart Confusion Matrix
confusionMatrix(yhat_rpart_all, reference = test2$is_fraud)
# Tabulate Confusion Matrix table
confusionMatrix(yhat_rpart_all, reference = test2$is_fraud)$table

# Calculating Model Costs
cost_preds2 <- cbind(cost_preds, yhat_rpart_all)
rp_all_saved <- cost_preds2 %>%
  filter(results == 1 & yhat_rpart_all == 1) %>%
  summarize(sum(amt)) %>%
  pull()
rp_all_missed <- cost_preds2 %>%
  filter(results == 1 & yhat_rpart_all == 0) %>%
  summarize(sum(amt)) %>%
  pull()
rp_all_misclassified <- cost_preds2 %>%
  filter(results == 0 & yhat_rpart_all == 1) %>%
  summarize(sum(amt)) %>%
  pull()
rp_all_savedpct <- rp_all_saved/cost
rp_all_misclassifiedpct <- rp_all_misclassified/cost

# Evaluating model performance 
rp_all_specificity <- confusionMatrix(yhat_rpart_all, 
                                      reference = test2$is_fraud)$byClass[["Specificity"]]
rp_all_NPV <- confusionMatrix(yhat_rpart_all, 
                              reference = test2$is_fraud)$byClass[["Neg Pred Value"]]

# Saving results in a table
cost_results <- bind_rows(cost_results,
                          data.frame(Model = "Rpart Model 1: All Variables",
                                 AmtSaved = rp_all_saved,
                                 FraudMissed = rp_all_missed,
                                 Misclassified = rp_all_misclassified,
                                 SavedPct = rp_all_savedpct,
                                 MisclassPct = rp_all_misclassifiedpct,
                                 Specificity = rp_all_specificity,
                                 NPV = rp_all_NPV))

# Variable importance
# Merchant highest score
fit_rpart_all$variable.importance 

##### Rpart Model 2 #####

# Basic model with only amount, category and date parts variables
fit_rpart <- train2 %>%
  select(-c(trans_date_trans_time, cc_num, merchant, bins)) %>%
  rpart(is_fraud ~ ., data = ., method = "class")

# Rpart Model 2 predictions
yhat_rpart <- predict(fit_rpart, test2, type = "class")

# Rpart Model 2 Confusion Matrix
confusionMatrix(yhat_rpart, reference = test2$is_fraud)

# Tabulate Confusion Matrix table
confusionMatrix(yhat_rpart, reference = test2$is_fraud)$table

# Calculating Model 2 Costs
cost_preds3 <- cbind(cost_preds2, yhat_rpart)
rp_saved <- cost_preds3 %>%
  filter(results == 1 & yhat_rpart == 1) %>%
  summarize(sum(amt)) %>%
  pull()
rp_missed <- cost_preds3 %>%
  filter(results == 1 & yhat_rpart == 0) %>%
  summarize(sum(amt)) %>%
  pull()
rp_misclassified <- cost_preds3 %>%
  filter(results == 0 & yhat_rpart == 1) %>%
  summarize(sum(amt)) %>%
  pull()
rp_savedpct = rp_saved/cost
rp_misclassifiedpct = rp_misclassified/cost
  
# Evaluating model 2 performance 
rp_specificity <- confusionMatrix(yhat_rpart, 
                                  reference = test2$is_fraud)$byClass[["Specificity"]]
rp_NPV <- confusionMatrix(yhat_rpart, 
                          reference = test2$is_fraud)$byClass[["Neg Pred Value"]]

# Saving results in a table
cost_results <- bind_rows(cost_results,
                          data.frame(Model = "Rpart Model 2: Amount, Category & Date Variables",
                                     AmtSaved = rp_saved,
                                     FraudMissed = rp_missed,
                                     Misclassified = rp_misclassified,
                                     SavedPct = rp_savedpct,
                                     MisclassPct = rp_misclassifiedpct,
                                     Specificity = rp_specificity,
                                     NPV = rp_NPV))

# Plotting cp vs cross validation error
as.data.frame(fit_rpart$cptable) %>%
  ggplot(aes(x = CP,y = xerror)) +
  geom_point(alpha = 0.5) + 
  geom_line(colour = "darkorange2") + 
  ylim(0.35,1) + 
  scale_x_reverse() + 
  labs(x = "Complexity Parameter", y = "Cross-validation Error") + 
  theme_bw(base_size = 10) +
  theme(panel.grid.minor = element_blank(),
        panel.grid.major.y = element_blank())

#####  Rpart Model 2: Tuning Complexity Parameter (CP) #####

# Basic rpart model with minsplit & cp = 0  
fit_rpartcp <- train2 %>%
  select(-c(trans_date_trans_time, cc_num, merchant, bins)) %>%
  rpart(is_fraud ~ ., data = ., minsplit = 0, cp = 0, method = "class")

# Plotting cp/xerror
plotcp(fit_rpartcp)

as.data.frame(fit_rpartcp$cptable) %>%
  ggplot(aes(x = CP,y = xerror)) +
  geom_point(alpha = 0.5) + 
  geom_line(color = "darkorange2") +  
  scale_x_reverse() + 
  labs(x = "Complexity Parameter",y = "Cross-validation Error") + 
  theme_bw(base_size = 10) +
  theme(panel.grid.minor = element_blank(),
        panel.grid.major.y = element_blank())

cptable <- fit_rpartcp$cptable
n <- cptable[which.min(cptable[,"xerror"]),][["nsplit"]]

# Choosing cp with fewer splits 
rp_cp <- as.data.frame(fit_rpartcp$cptable) %>% 
  filter(nsplit == n) %>% 
  pull(CP)

# Pruning basic rpart model   
pfit_rpartcp <- prune.rpart(fit_rpartcp, cp = rp_cp)

# Rpart Model 2 cp predictions
yhat_rpartcp <- predict(pfit_rpartcp, test2, type = "class")

# Rpart Model 2 cp Confusion Matrix
confusionMatrix(yhat_rpartcp, reference = test2$is_fraud)
# Tabulate Confusion Matrix table
confusionMatrix(yhat_rpartcp, reference = test2$is_fraud)$table

# Tabulate Rpart Model 2 cp Confusion Matrix
cm_rpcp <- as.data.frame.matrix(confusionMatrix(yhat_rpartcp, test2$is_fraud)$table)

# Tabulate Rpart Model 2 Confusion Matrix
cm_rp <- as.data.frame.matrix(confusionMatrix(yhat_rpart, test2$is_fraud)$table)

# Combining the two confusion matrices
combined <- cbind(cm_rp, cm_rpcp)

# Calculating Model 2 Tuned CP Costs
cost_preds4 <- cbind(cost_preds3, yhat_rpartcp)
rpcp_saved <- cost_preds4 %>%
  filter(results == 1 & yhat_rpartcp == 1) %>%
  summarize(sum(amt)) %>%
  pull()
rpcp_missed <- cost_preds4 %>%
  filter(results == 1 & yhat_rpartcp == 0) %>%
  summarize(sum(amt)) %>%
  pull()
rpcp_misclassified <- cost_preds4 %>%
  filter(results == 0 & yhat_rpartcp == 1) %>%
  summarize(sum(amt)) %>%
  pull()
rpcp_savedpct <- rpcp_saved/cost
rpcp_misclassifiedpct <- rpcp_misclassified/cost

# Evaluating model 2 cp performance 
rpcp_specificity <- confusionMatrix(yhat_rpartcp, 
                                    reference = test2$is_fraud)$byClass[["Specificity"]]
rpcp_NPV <- confusionMatrix(yhat_rpartcp, 
                            reference = test2$is_fraud)$byClass[["Neg Pred Value"]]

# Saving results in a table
cost_results <- bind_rows(cost_results,
                          data.frame(Model = "Rpart Model 2: Tuned CP",
                                     AmtSaved = rpcp_saved,
                                     FraudMissed = rpcp_missed,
                                     Misclassified = rpcp_misclassified,
                                     SavedPct = rpcp_savedpct,
                                     MisclassPct = rpcp_misclassifiedpct,
                                     Specificity = rpcp_specificity,
                                     NPV = rpcp_NPV))

##### Rpart Model 3 #####

# Verifying cc_num as predictor. no merchant, bins, trans_date\time
fit_rpart_cc <- train2 %>%
  select(-c(trans_date_trans_time, merchant, bins)) %>%
  rpart(is_fraud ~ ., data = ., method = "class")

# Rpart Model 3 predictions
yhat_rpart_cc <- predict(fit_rpart_cc, test2, type = "class")

# Confusion Matrix for Rpart Model 3 CC
cm_rpcc <- as.data.frame.matrix(confusionMatrix(yhat_rpart_cc, test2$is_fraud)$table)

# Combining confusion matrices
combined2 <-cbind(cm_rp, cm_rpcc)

# Calculating Rpart Model 3 costs
cost_preds5 <- cbind(cost_preds4, yhat_rpart_cc)
rpcc_saved <- cost_preds5 %>%
  filter(results == 1 & yhat_rpart_cc == 1) %>%
  summarize(sum(amt)) %>%
  pull()
rpcc_missed <- cost_preds5 %>%
  filter(results == 1 & yhat_rpart_cc == 0) %>%
  summarize(sum(amt)) %>%
  pull()
rpcc_misclassified <- cost_preds5 %>%
  filter(results == 0 & yhat_rpart_cc == 1) %>%
  summarize(sum(amt)) %>%
  pull()
rpcc_savedpct <- rpcc_saved/cost
rpcc_misclassifiedpct <- rpcc_misclassified/cost

# Evaluating Rpart Model 3 performance
rpcc_specificity <- confusionMatrix(yhat_rpart_cc, 
                                    reference = test2$is_fraud)$byClass[["Specificity"]]
rpcc_NPV <- confusionMatrix(yhat_rpart_cc, 
                            reference = test2$is_fraud)$byClass[["Neg Pred Value"]]

# Saving results in a table
cost_results <- bind_rows(cost_results,
                          data.frame(Model = "Rpart Model 3: CC",
                                     AmtSaved = rpcc_saved,
                                     FraudMissed = rpcc_missed,
                                     Misclassified = rpcc_misclassified,
                                     SavedPct = rpcc_savedpct,
                                     MisclassPct = rpcc_misclassifiedpct,
                                     Specificity = rpcc_specificity,
                                     NPV = rpcc_NPV))

# Plotting cp/xerror
plotcp(fit_rpart_cc)

as.data.frame(fit_rpart_cc$cptable) %>%
  ggplot(aes(x = CP, y = xerror)) +
  geom_point(alpha = 0.5) +
  geom_line(colour = "darkorange2") +
  scale_x_reverse() + 
  labs(x = "Complexity Parameter",y = "Cross-validation Error") + 
  theme_bw(base_size = 10) +
  theme(panel.grid.minor = element_blank(),
        panel.grid.major.y = element_blank())

#####  Rpart Model 3 CC: Tuning Complexity Parameter (CP) #####

# Rpart Model 3 CC tuning cp to verify cc_num as predictor. no merchant, bins, trans_date\time
fit_rpartcp_cc <- train2 %>% select(-c(trans_date_trans_time, merchant, bins)) %>% 
  rpart(is_fraud ~ ., data = ., minsplit = 0, cp = 0, method = "class")

# Plotting cp/err
as.data.frame(fit_rpartcp_cc$cptable) %>%
  ggplot(aes(x = CP, y = xerror)) +
  geom_point(alpha = 0.5) +
  geom_line(colour = "darkorange2") +
  scale_x_reverse() + 
  labs(x = "Complexity Parameter",y = "Cross-validation Error") + 
  theme_bw(base_size = 10) +
  theme(panel.grid.minor = element_blank(),
        panel.grid.major.y = element_blank())

cc_cptable <- fit_rpartcp_cc$cptable
cc_n <- cc_cptable[which.min(cc_cptable[,"xerror"]),][["nsplit"]]

# Choosing cp with fewer splits 
rpcc_cp <- as.data.frame(cc_cptable) %>% 
  filter(nsplit == cc_n) %>% 
  pull(CP)

# Pruning Rpart Model 3 
pfit_rpartcc_cp <- prune.rpart(fit_rpartcp_cc, cp = rpcc_cp)

# Rpart Model 3 cp predictions
yhat_rpartcc_cp <- predict(pfit_rpartcc_cp, test2, type = "class")

# Tabulate Rpart Model 3 CC CP Confusion Matrix
cm_rpcc_cp <- as.data.frame.matrix(confusionMatrix(yhat_rpartcc_cp, test2$is_fraud)$table)

# Combining Confusion Matrices
combined3 <- cbind(cm_rpcc, cm_rpcc_cp)

# Calculating Rpart Model 3 CP Tuned costs
cost_preds6 <- cbind(cost_preds5, yhat_rpartcc_cp)
rpcc_cpsaved <- cost_preds6 %>%
  filter(results == 1 & yhat_rpartcc_cp == 1) %>%
  summarize(sum(amt)) %>%
  pull()
rpcc_cpmissed <- cost_preds6 %>%
  filter(results == 1 & yhat_rpartcc_cp == 0) %>%
  summarize(sum(amt)) %>%
  pull()
rpcc_cpmisclassified <- cost_preds6 %>%
  filter(results == 0 & yhat_rpartcc_cp == 1) %>%
  summarize(sum(amt)) %>%
  pull()
rpcc_cpsavedpct <- rpcc_cpsaved/cost
rpcc_cpmisclassifiedpct <- rpcc_cpmisclassified/cost

# Evaluating Rpart Model 3 CP Tuned performance
rpcc_cp_specificity <- confusionMatrix(yhat_rpartcc_cp, 
                                       reference = test2$is_fraud)$byClass[["Specificity"]]
rpcc_cp_NPV <- confusionMatrix(yhat_rpartcc_cp, 
                            reference = test2$is_fraud)$byClass[["Neg Pred Value"]]

# Saving results in a table
cost_results <- bind_rows(cost_results,
                          data.frame(Model = "Rpart Model 3: CC Tuned CP",
                                     AmtSaved = rpcc_cpsaved,
                                     FraudMissed = rpcc_cpmissed,
                                     Misclassified = rpcc_cpmisclassified,
                                     SavedPct = rpcc_cpsavedpct,
                                     MisclassPct = rpcc_cpmisclassifiedpct,
                                     Specificity = rpcc_cp_specificity,
                                     NPV = rpcc_cp_NPV))

##### Final Validation Set Up #####

# Tabulating results & amounts
final_preds <- tibble(amt = test_set$amt, results = test_set$is_fraud)

# Tabulating final cost results
final_results <- tibble(Model = "No Fraud Predicted", 
                        AmtSaved = 0, 
                        FraudMissed = cost, 
                        Misclassified = 0, 
                        SavedPct = 0, 
                        MisclassPct = 0, 
                        Specificity = 0, 
                        NPV = 0)

##### Final Validation: Rpart Model 2 Tuned CP #####

# Rpart Model 2 cp predictions
yhat_rpartcp <- predict(pfit_rpartcp, test_set, type = "class")

# Tabulate Rpart Model 2 CP Confusion Matrix
cm_rp <- as.data.frame.matrix(confusionMatrix(yhat_rpartcp, 
                                              reference = test_set$is_fraud)$table)

# Calculating Rpart Model 2 CP Tuned costs
final_preds <- cbind(final_preds, yhat_rpartcp)
rpcp_saved <- final_preds %>%
  filter(results == 1 & yhat_rpartcp == 1) %>%
  summarize(sum(amt)) %>%
  pull()
rpcp_missed <- final_preds %>%
  filter(results == 1 & yhat_rpartcp == 0) %>%
  summarize(sum(amt)) %>%
  pull()
rpcp_misclassified <- final_preds %>%
  filter(results == 0 & yhat_rpartcp == 1) %>%
  summarize(sum(amt)) %>%
  pull()
rpcp_savedpct <- rpcp_saved/cost2
rpcp_misclassifiedpct <- rpcp_misclassified/cost2

# Evaluating model performance
rpcp_specificity <- confusionMatrix(yhat_rpartcp, 
                                    reference = test_set$is_fraud)$byClass[["Specificity"]]
rpcp_NPV <- confusionMatrix(yhat_rpartcp, 
                            reference = test_set$is_fraud)$byClass[["Neg Pred Value"]]

# Saving results in final table
final_results <- bind_rows(final_results,
                           data.frame(Model = "Rpart Model 2: Tuned CP",
                                      AmtSaved = rpcp_saved,
                                      FraudMissed = rpcp_missed,
                                      Misclassified = rpcp_misclassified,
                                      SavedPct = rpcp_savedpct,
                                      MisclassPct = rpcp_misclassifiedpct,
                                      Specificity = rpcp_specificity,
                                      NPV = rpcp_NPV))

cost_preds <- cost_preds6

# removing rpart variables
rm(rp_all_saved, rp_all_missed, rp_all_misclassified, rp_all_savedpct, rp_all_misclassifiedpct,
   rp_all_specificity, rp_all_NPV, rp_saved, rp_missed, rp_misclassified, rp_savedpct, 
   rp_misclassifiedpct, rp_specificity, rp_NPV, rpcc_saved, rpcc_missed, rpcc_misclassified, 
   rpcc_savedpct, rpcc_misclassifiedpct, rpcc_specificity, rpcc_NPV, rpcp_saved, rpcp_missed, 
   rpcp_misclassified, rpcp_savedpct, rpcp_misclassifiedpct, rpcp_specificity, rpcp_NPV, 
   rpcc_cpsaved, rpcc_cpmissed, rpcc_cpmisclassified, rpcc_cpsavedpct, rpcc_cpmisclassifiedpct, 
   rpcc_cp_specificity, rpcc_cp_NPV,
   yhat_rpart, yhat_rpart_all, yhat_rpart_cc, yhat_rpartcc_cp, yhat_rpartcp,
   cm_rpcp, cm_rpcc, cm_rpcc_cp,
   cost_preds2, cost_preds3, cost_preds4, cost_preds5, cost_preds6,
   cc_cptable, cptable, cm_df, n, cc_n, rp_cp, rpcc_cp, cc_number, Actual, Predicted, Y)

# Removing rpart models to clear space
rm(fit_rpart,
   fit_rpart_all,
   fit_rpart_cc, 
   fit_rpartcp,
   fit_rpartcp_cc,
   pfit_rpartcp,
   pfit_rpartcc_cp)

##### Logistic Regression #####

## GLM Model 1 ##

# GLM Model 1 fits amount & category interaction and date variables as factors
fit_glm1 <- train2 %>% 
  glm(is_fraud ~ category * amt + hour + day + month + weekday, data = ., 
      family = "binomial")

# GLM Model 1 predictions
phat_glm1 <- predict.glm(fit_glm1, test2, type = "response")
yhat_glm1 <- ifelse(phat_glm1 > 0.5, 1, 0) %>% 
  factor()

varImp(fit_glm1)

## GLM Model 2 ##

# GLM model 2 fits category & amt interaction, no bins, date parts as factors
fit_glm2 <- train2 %>% 
  glm(is_fraud ~ category + amt + bins + hour + day + month + weekday, data = ., 
      family = "binomial")

# GLM Model 2 predictions
phat_glm2 <- predict.glm(fit_glm2, test2, type = "response")
yhat_glm2 <- ifelse(phat_glm2 > 0.5, 1, 0) %>% 
  factor()

varImp(fit_glm2)

# Confusion Matrix GLM Model 1 & 2
cm_glm1 <- confusionMatrix(yhat_glm1, reference = test2$is_fraud)$table
cm_glm2 <- confusionMatrix(yhat_glm2, reference = test2$is_fraud)$table

combined4 <- cbind(cm_glm1, cm_glm2)

rm(cm_glm1, cm_glm2)

# Evaluating GLM Models 1 & 2 performance
glm1_specificity <- confusionMatrix(yhat_glm1, 
                                    reference = test2$is_fraud)$byClass[["Specificity"]]
glm1_NPV <- confusionMatrix(yhat_glm1, 
                            reference = test2$is_fraud)$byClass[["Neg Pred Value"]]

glm2_specificity <- confusionMatrix(yhat_glm2, 
                                    reference = test2$is_fraud)$byClass[["Specificity"]]
glm2_NPV <- confusionMatrix(yhat_glm2, 
                            reference = test2$is_fraud)$byClass[["Neg Pred Value"]]

# Calculating GLM Model 1 Costs
cost_preds <- cbind(cost_preds, yhat_glm1, yhat_glm2)
glm1_saved <- cost_preds %>%
  filter(results == 1 & yhat_glm1 == 1) %>%
  summarize(sum(amt)) %>%
  pull()
glm1_missed <- cost_preds %>%
  filter(results == 1 & yhat_glm1 == 0) %>%
  summarize(sum(amt)) %>%
  pull()
glm1_misclassified <- cost_preds %>%
  filter(results == 0 & yhat_glm1 == 1) %>%
  summarize(sum(amt)) %>%
  pull()
glm1_savedpct <- glm1_saved/cost
glm1_misclassifiedpct <- glm1_misclassified/cost

# Calculating GLM Model 2 Costs
glm2_saved <- cost_preds %>%
  filter(results == 1 & yhat_glm2 == 1) %>%
  summarize(sum(amt)) %>%
  pull()
glm2_missed <- cost_preds %>%
  filter(results == 1 & yhat_glm2 == 0) %>%
  summarize(sum(amt)) %>%
  pull()
glm2_misclassified <- cost_preds %>%
  filter(results == 0 & yhat_glm2 == 1) %>%
  summarize(sum(amt)) %>%
  pull()
glm2_savedpct <- glm2_saved/cost
glm2_misclassifiedpct <- glm2_misclassified/cost

# Saving results to table
cost_results <- bind_rows(cost_results,
                          data.frame(Model = "GLM Model 1: category * amount",
                                     AmtSaved = glm1_saved,
                                     FraudMissed = glm1_missed,
                                     Misclassified = glm1_misclassified,
                                     SavedPct = glm1_savedpct,
                                     MisclassPct = glm1_misclassifiedpct,
                                     Specificity = glm1_specificity,
                                     NPV = glm1_NPV))
cost_results <- bind_rows(cost_results,
                          data.frame(Model = "GLM Model 2: category + amount",
                                     AmtSaved = glm2_saved,
                                     FraudMissed = glm2_missed,
                                     Misclassified = glm2_misclassified,
                                     SavedPct = glm2_savedpct,
                                     MisclassPct = glm2_misclassifiedpct,
                                     Specificity = glm2_specificity,
                                     NPV = glm2_NPV))

# Print Cost Results GLM Model 1 & 2
cost_results %>% 
  filter(Model == c("GLM Model 1: category * amount",
                    "GLM Model 2: category + amount"))

# Removing large elements 
rm(glm1_saved, glm1_missed, glm1_misclassified, glm1_savedpct, glm1_misclassifiedpct, 
   glm1_specificity, glm1_NPV, glm2_saved, glm2_missed, glm2_misclassified, glm2_savedpct,
   glm2_misclassifiedpct, glm2_specificity, glm2_NPV, phat_glm1, phat_glm2, yhat_glm1,
   yhat_glm2, fit_glm1, fit_glm2)

## GLM Model 3 ##

##### Loading Generated Models #####

# If you want to run the code, but not spend the time processing the models:
# This will download fit models and load them into your RStudio environment.

# create temp file & url 
dl <- tempfile()
URL <- "https://github.com/Wilsven/Credit-Card-Fraud-Detection/releases/download/v1-files/CYO_Models.RData"

# download url into temp file
download.file(URL, dl)

# load .RData into environment. This may take a few minutes.
load(dl)

# Removing large elements
rm(dl, URL, fit_glm1, fit_glm2, fit_glm3, fit_rf51)

# Model fitted with category, amount and bins interactions, including all date variables
# fit_glm3 <- train2 %>%
# glm(is_fraud ~ category * amt * bins + hour + day + month + weekday, data = ., 
#    family = "binomial", model = FALSE, y = FALSE)
fit_glm3 <- fit_glm_bins

# GLM Model 3 estimates
test2_temp <- test2 %>%
  rename(trans_hour = hour)
phat_glm3 <- predict.glm(fit_glm3, test2_temp, type = "response")

# GLM Model 3 predictions at 0.5
yhat_glm3_.5 <- ifelse(phat_glm3 > 0.5, 1, 0) %>% 
  factor()
# GLM Model 3 predictions at 0.4
yhat_glm3_.4 <- ifelse(phat_glm3 > 0.4, 1, 0) %>% 
  factor()

# GLM Model 3 Confusion Matrices
cm_glm3_.5 <- as.data.frame.matrix(confusionMatrix(yhat_glm3_.5, 
                                                   reference = test2$is_fraud)$table)
cm_glm3_.4 <- as.data.frame.matrix(confusionMatrix(yhat_glm3_.4, 
                                                   reference = test2$is_fraud)$table)

combined5 <- cbind(cm_glm3_.5, cm_glm3_.4)

rm(cm_glm3_.5, cm_glm3_.4)

# Evaluating GLM Model 3 performance
glm3_.5_specificity <- confusionMatrix(yhat_glm3_.5, 
                                       reference = test2$is_fraud)$byClass[["Specificity"]]
glm3_.4_specificity <- confusionMatrix(yhat_glm3_.4, 
                                       reference = test2$is_fraud)$byClass[["Specificity"]]
glm3_.5_NPV <- confusionMatrix(yhat_glm3_.5, 
                               reference = test2$is_fraud)$byClass[["Neg Pred Value"]]
glm3_.4_NPV <- confusionMatrix(yhat_glm3_.4, 
                               reference = test2$is_fraud)$byClass[["Neg Pred Value"]]

cost_preds <- cbind(cost_preds, yhat_glm3_.5, yhat_glm3_.4)

# Calculating GLM Model 3 > 0.5 Costs
glm3_.5_saved <- cost_preds %>%
  filter(results == 1 & yhat_glm3_.5 == 1) %>%
  summarize(sum(amt)) %>%
  pull()
glm3_.5_missed <- cost_preds %>%
  filter(results == 1 & yhat_glm3_.5 == 0) %>%
  summarize(sum(amt)) %>%
  pull()
glm3_.5_misclassified <- cost_preds %>%
  filter(results == 0 & yhat_glm3_.5 == 1) %>%
  summarize(sum(amt)) %>%
  pull()
glm3_.5_savedpct <- glm3_.5_saved/cost
glm3_.5_misclassifiedpct <- glm3_.5_misclassified/cost

# Calculating GLM Model 3 > 0.4 Costs
glm3_.4_saved <- cost_preds %>%
  filter(results == 1 & yhat_glm3_.4 == 1) %>%
  summarize(sum(amt)) %>%
  pull()
glm3_.4_missed <- cost_preds %>%
  filter(results == 1 & yhat_glm3_.4 == 0) %>%
  summarize(sum(amt)) %>%
  pull()
glm3_.4_misclassified <- cost_preds %>%
  filter(results == 0 & yhat_glm3_.4 == 1) %>%
  summarize(sum(amt)) %>%
  pull()
glm3_.4_savedpct <- glm3_.4_saved/cost
glm3_.4_misclassifiedpct <- glm3_.4_misclassified/cost

# Saving results to table
cost_results <- bind_rows(cost_results,
                          data.frame(Model = "GLM Model 3: > 0.5",
                                     AmtSaved = glm3_.5_saved,
                                     FraudMissed = glm3_.5_missed,
                                     Misclassified = glm3_.5_misclassified,
                                     SavedPct = glm3_.5_savedpct,
                                     MisclassPct = glm3_.5_misclassifiedpct,
                                     Specificity = glm3_.5_specificity,
                                     NPV = glm3_.5_NPV))
cost_results <- bind_rows(cost_results,
                          data.frame(Model = "GLM Model 3: > 0.4",
                                     AmtSaved = glm3_.4_saved,
                                     FraudMissed = glm3_.4_missed,
                                     Misclassified = glm3_.4_misclassified,
                                     SavedPct = glm3_.4_savedpct,
                                     MisclassPct = glm3_.4_misclassifiedpct,
                                     Specificity = glm3_.4_specificity,
                                     NPV = glm3_.4_NPV))

# Print Cost Results GLM Model 3 
cost_results %>% 
  filter(Model == c("GLM Model 3: > 0.5",
                    "GLM Model 3: > 0.4"))

# Removing large elements
rm(phat_glm3, yhat_glm3_.5, yhat_glm3_.4)

##### Final Validation: GLM Model 3 #####

# Predictions
test_set_temp <- test_set %>%
  rename(trans_hour = hour)
phat_glm3 <- predict.glm(fit_glm3, test_set_temp, type = "response")
yhat_glm3_.4 <- ifelse(phat_glm3 > 0.4, 1, 0) %>%
  factor()

# Evaluating GLM Model 3 performance
cm_glm <- as.data.frame.matrix(confusionMatrix(yhat_glm3_.4, 
                                               reference = test_set$is_fraud)$table)
glm3_.4_specificity <- confusionMatrix(yhat_glm3_.4, 
                                       reference = test_set$is_fraud)$byClass[["Specificity"]]
glm3_.4_NPV <- confusionMatrix(yhat_glm3_.4, 
                               reference = test_set$is_fraud)$byClass[["Neg Pred Value"]]

# Calculating GLM Model 3 > 0.4 Costs
final_preds <- cbind(final_preds, yhat_glm3_.4)
glm3_.4_saved <- final_preds %>%
  filter(results == 1 & yhat_glm3_.4 == 1) %>%
  summarize(sum(amt)) %>%
  pull()
glm3_.4_missed <- final_preds %>%
  filter(results == 1 & yhat_glm3_.4 == 0) %>%
  summarize(sum(amt)) %>%
  pull()
glm3_.4_misclassified <- final_preds %>%
  filter(results == 0 & yhat_glm3_.4 == 1) %>%
  summarize(sum(amt)) %>%
  pull()
glm3_.4_savedpct <- glm3_.4_saved/cost2
glm3_.4_misclassifiedpct <- glm3_.4_misclassified/cost2

# Saving results
final_results <- bind_rows(final_results,
                           data.frame(Model = "GLM Model 3: > 0.4",
                                      AmtSaved = glm3_.4_saved,
                                      FraudMissed = glm3_.4_missed,
                                      Misclassified = glm3_.4_misclassified,
                                      SavedPct = glm3_.4_savedpct,
                                      MisclassPct = glm3_.4_misclassifiedpct,
                                      Specificity = glm3_.4_specificity,
                                      NPV = glm3_.4_NPV))

# Removing GLM saved values
rm(glm3_.4_saved, glm3_.4_missed, glm3_.4_misclassified, glm3_.4_savedpct, 
   glm3_.4_misclassifiedpct, glm3_.4_specificity, glm3_.4_NPV, glm3_.5_saved, 
   glm3_.5_missed, glm3_.5_misclassified, glm3_.5_savedpct, 
   glm3_.5_misclassifiedpct, glm3_.5_specificity, glm3_.5_NPV)

# Removing large elements
rm(fit_glm3, fit_glm_bins, test2_temp, yhat_glm3_.4, phat_glm3, 
   combined, combined2, combined3, combined4, combined5)

##### Random Forest #####

## Random Forest Model 1 ##

# Resetting seed
set.seed(1, sample.kind = "Rounding")

# Compute cost of not detecting fraud
cost <- sum(test2$amt[test2$is_fraud == 1])

# Random Forest Model 1 
fit_rf <- train2 %>% 
  randomForest(is_fraud ~ ., data = .,
               ntree = 51, 
               replacement = TRUE,
               importance = TRUE)

# Random Forest Model 1 predictions
yhat_rf <- predict(fit_rf, test2, type = "class")

# Evaluating Random Forest Model 1 performance
confusionMatrix(yhat_rf, reference = test2$is_fraud)$table
rf_specificity <- confusionMatrix(yhat_rf, 
                                 reference = test2$is_fraud)$byClass[["Specificity"]]
rf_NPV <- confusionMatrix(yhat_rf, 
                          reference = test2$is_fraud)$byClass[["Neg Pred Value"]]

# Calculating Random Forest Model 1 Costs
cost_preds <- cbind(cost_preds, yhat_rf)
rf_saved <- cost_preds %>%
  filter(results == 1 & yhat_rf == 1) %>%
  summarize(sum(amt)) %>%
  pull()
rf_missed <- cost_preds %>%
  filter(results == 1 & yhat_rf == 0) %>%
  summarize(sum(amt)) %>%
  pull()
rf_misclassified <- cost_preds %>%
  filter(results == 0 & yhat_rf == 1) %>%
  summarize(sum(amt)) %>%
  pull()
rf_savedpct <- rf_saved/cost
rf_misclassifiedpct <- rf_misclassified/cost

# Saving results
cost_results <- bind_rows(cost_results,
                          data.frame(Model = "Random Forest Model 1",
                                     AmtSaved = rf_saved,
                                     FraudMissed = rf_missed,
                                     Misclassified = rf_misclassified,
                                     SavedPct = rf_savedpct,
                                     MisclassPct = rf_misclassifiedpct,
                                     Specificity = rf_specificity,
                                     NPV = rf_NPV))
cost_results %>% 
  filter(Model == "Random Forest Model 1")

# Plotting trees / error
plot(fit_rf)

# min OOB & fraud class Error at mtry 3
oobData <- as.data.table(plot(fit_rf))
min(oobData$OOB)
min(oobData$'1')

# Add new trees column into OOB data table
oobData[, trees := .I]

# Cast to long format
oobData <- melt(oobData, id.vars = "trees")
setnames(oobData, "value", "error")

# Plotting trees / error
oobData %>% 
  ggplot(aes(x = trees, y = error)) + 
  geom_line(colour = "darkorange2") + 
  facet_wrap(~variable, scales = "free") +
  labs(title = "Out-of-Bag & Class Errors by Number of Trees") +
  theme_bw(base_size = 10) + 
  theme(plot.title = element_text(size = 9))

# Variable importance
fit_rf$importance

## Tuning RF Model for best mtry value ##

# Tuning rf for mtry 4:6, 75 trees
rf_tune <-  tuneRF(train2[-6], train2$is_fraud, 
                   mtryStart = 5, 
                   ntreeTry = 75, 
                   stepFactor = 0.9,
                   trace = TRUE, 
                   plot = TRUE)

options(digits = 6)
rf_tune

as.data.frame(rf_tune) %>% 
  mutate(OOBError = OOBError*100) %>%
  ggplot(aes(x = mtry, y = OOBError)) +
  geom_point() + 
  geom_line(colour = "#56B4E9") + 
  ylim(0.2, 0.21) +
  geom_text(aes(label=sprintf("%1.3f%%",OOBError)), size = 2.5,  nudge_y = .001,alpha =3/4) +
  scale_x_continuous(breaks=c(4,5,6))+ theme_bw(base_size = 10) + 
  theme(plot.title = element_text(size = 9)) + 
  ggtitle("OOB Error Rate by Mtry Value")

# Removing large elements
rm(fit_rf, oobData, rf_saved, rf_missed, rf_misclassified, 
   rf_savedpct, rf_misclassifiedpct, rf_specificity, rf_NPV, yhat_rf, rf_tune)

## Random Forest Model 2 ##

# 251 trees, 4/10 predictors
# fit_rf251 <- train2 %>% 
#  randomForest(is_fraud ~., data = .,
#               ntree = 251, mtry = 4,
#               replacement = TRUE,
#               importance = TRUE)
fit_rf2 <- fit_rf251

# Random Forest Model 2 predictions 
test2_temp <- test2 %>%
  rename(trans_hour = hour)
yhat_rf2 <- predict(fit_rf2, test2_temp, type = "class")

# Evaluating model performance
confusionMatrix(yhat_rf2, reference = test2_temp$is_fraud)$table
rf2_specificity <- confusionMatrix(yhat_rf2, 
                                   reference = test2_temp$is_fraud)$byClass[["Specificity"]]
rf2_NPV <- confusionMatrix(yhat_rf2, 
                           reference = test2_temp$is_fraud)$byClass[["Neg Pred Value"]]

# Calculating Random Forest Model 2 Costs
cost_preds <- cbind(cost_preds, yhat_rf2)
rf2_saved <- cost_preds %>%
  filter(results == 1 & yhat_rf2 == 1) %>%
  summarize(sum(amt)) %>%
  pull()
rf2_missed <- cost_preds %>%
  filter(results == 1 & yhat_rf2 == 0) %>%
  summarize(sum(amt)) %>%
  pull()
rf2_misclassified <- cost_preds %>%
  filter(results == 0 & yhat_rf2 == 1) %>%
  summarize(sum(amt)) %>%
  pull()
rf2_savedpct <- rf2_saved/cost
rf2_misclassifiedpct <- rf2_misclassified/cost

# Saving results
cost_results <- bind_rows(cost_results,
                          data.frame(Model = "Random Forest Model 2",
                                     AmtSaved = rf2_saved,
                                     FraudMissed = rf2_missed,
                                     Misclassified = rf2_misclassified,
                                     SavedPct = rf2_savedpct,
                                     MisclassPct = rf2_misclassifiedpct,
                                     Specificity = rf2_specificity,
                                     NPV = rf2_NPV))

cost_results %>% 
  filter(Model == "Random Forest Model 2")

# Plot the model
plot(fit_rf2)

# Get OOB data from plot and coerce to data.table
oobData <- as.data.table(plot(fit_rf2))

# Define trees as 1:ntree
oobData[, trees := .I]

# Cast to long format
oobData <- melt(oobData, id.vars = "trees")
setnames(oobData, "value", "error")

# Plotting trees / error
oobData %>% 
  filter(variable == 1) %>%
  ggplot(aes(x = trees, y = error)) + 
  geom_line(colour = "darkorange2") + 
  ylim(0.26, 0.325) +
  labs(title = "Fraud Error Rate by Number of Trees") +
  theme_bw(base_size = 10) + 
  theme(plot.title = element_text(size = 9))

##### Final Validation: RF Model 2 #####

# Random Forest Model 2 predictions 
test_set_temp <- test_set %>%
  rename(trans_hour = hour)
yhat_rf2 <- predict(fit_rf2, test_set_temp, type = "class")

# Evaluating model performance
cm_rf <- as.data.frame.matrix(confusionMatrix(yhat_rf2, 
                                              reference = test_set_temp$is_fraud)$table)
rf2_specificity <- confusionMatrix(yhat_rf2, 
                                   reference = test_set_temp$is_fraud)$byClass[["Specificity"]]
rf2_NPV <- confusionMatrix(yhat_rf2, 
                           reference = test_set_temp$is_fraud)$byClass[["Neg Pred Value"]]

# Calculating Random Forest Model 2 Costs
final_preds <- cbind(final_preds, yhat_rf2)
rf2_saved <- final_preds %>%
  filter(results == 1 & yhat_rf2 == 1) %>%
  summarize(sum(amt)) %>%
  pull()
rf2_missed <- final_preds %>%
  filter(results == 1 & yhat_rf2 == 0) %>%
  summarize(sum(amt)) %>%
  pull()
rf2_misclassified <- final_preds %>%
  filter(results == 0 & yhat_rf2 == 1) %>%
  summarize(sum(amt)) %>%
  pull()
rf2_savedpct <- rf2_saved/cost2
rf2_misclassifiedpct <- rf2_misclassified/cost2

# Saving results
final_results <- bind_rows(final_results,
                          data.frame(Model = "Random Forest Model 2",
                                     AmtSaved = rf2_saved,
                                     FraudMissed = rf2_missed,
                                     Misclassified = rf2_misclassified,
                                     SavedPct = rf2_savedpct,
                                     MisclassPct = rf2_misclassifiedpct,
                                     Specificity = rf2_specificity,
                                     NPV = rf2_NPV))

# Combining Confusion Matrices
cms <- cbind(cm_rp, cm_glm, cm_rf)

# Final Confusion Matrices
cms

# Final Cost Results Compared
final_results

# Removing large elements
rm(test2_temp, test_set_temp, fit_rf2, fit_rf251, oobData, yhat_rf2, rf2_saved, 
   rf2_missed, rf2_misclassified, rf2_savedpct, rf2_misclassifiedpct, rf2_specificity,
   rf2_NPV)