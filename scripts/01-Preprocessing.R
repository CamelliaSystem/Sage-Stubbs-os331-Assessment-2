################################################################################
# This is the code to carry out the research set out in my insurance fraud data#
# analysis report.                                                             #
################################################################################

install.packages("readxl")                                                      # Install required packages
install.packages("dplyr")                                                       # Install required packages
install.packages("stringr")                                                     # Install required packages
install.packages("caret")                                                       # Install required packages
install.packages("recipes")                                                     # Install required packages
install.packages("tidyverse")                                                   # Install required packages

library(readxl)                                                                 # Load required packages
library(dplyr)                                                                  # Load required packages
library(stringr)                                                                # Load required packages
library(caret)                                                                  # Load required packages
library(recipes)                                                                # Load required packages
library(tidyverse)                                                              # Load required packages

set.seed(123)                                                                   # Set seed for reproducability

fraud_data <- read_excel("fraud_data.xlsx")                                     # Load in data
attach(fraud_data)                                                              # Attach data

names(fraud_data)[names(fraud_data) == 'capital-gains']<-'capital_gains'
names(fraud_data)[names(fraud_data) == 'capital-loss']<-'capital_loss'
fraud_data<-fraud_data |>
  mutate(
    fraud_reported = factor(fraud_reported, levels = c("N","Y"))
  )                                                                             # Convert fraud reported variable to factor

fraud_data <- fraud_data |>
  mutate(
    across(where(is.character), ~ na_if(.x, "?"))
    )                                                                           # Convert missing values to NA

fraud_data$policy_bind_date<-as.Date(fraud_data$policy_bind_date,format="%d/%m/%Y") # Convert to date

fraud_data$incident_date<-as.Date(fraud_data$incident_date, format = "%d/%m/%Y")# Convert to date

fraud_data<-subset(fraud_data, select = -c(policy_number,insured_zip,incident_location)) # Remove unnecessary variables

idx<-createDataPartition(fraud_data$fraud_reported,p=0.8,list=FALSE)            # Split the data 80/20
train_data<-fraud_data[idx,]                                                    # Set train set
test_data<-fraud_data[-idx,]                                                    # Set test set

rec <- recipe(fraud_reported ~ ., data = train_data) |>
  step_date(all_date_predictors(), features = c("year", "month", "dow")) |>     # Convert date to numeric time since origin
  step_rm(all_date_predictors()) |>
  step_impute_mode(all_nominal_predictors()) |>                                 # Impute
  step_impute_median(all_numeric_predictors()) |>
  step_dummy(all_nominal_predictors(), one_hot = TRUE) |>                       # One-hot encode categoricals
  step_zv(all_predictors()) |>                                                  # Remove zero variance
  step_normalize(all_numeric_predictors())                                      # Scale numeric

prep_rec <- prep(rec, training = train_data, verbose = FALSE)

x_train <- bake(prep_rec, new_data = train_data) |> select(-fraud_reported)
y_train <- bake(prep_rec, new_data = train_data)$fraud_reported

x_test  <- bake(prep_rec, new_data = test_data) |> select(-fraud_reported)
y_test  <- bake(prep_rec, new_data = test_data)$fraud_reported

saveRDS(
  list(
    train_data = train_data, test_data = test_data,
    prep_rec = prep_rec,
    x_train = x_train, y_train = y_train,
    x_test  = x_test,  y_test  = y_test
  ),
  file = "preprocessed_data.rds"
)
