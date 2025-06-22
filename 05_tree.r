# Load required packages
library(caret)
library(rpart)
library(pROC)

# Variables selected from LASSO + manually added variables
selected_vars <- c("incm5", "DI1_pr", "DI2_pr", "HE_HTG", "BO1", "BE3_92", 
                   "HE_DMfh3", "HE_wc", "HE_glu", "HE_HDL_st2")

# Create analysis dataset
model_data <- data[, c("diabetes", selected_vars)]

# Remove missing values
model_data <- na.omit(model_data)

# Convert dependent variable to factor
model_data$diabetes <- factor(model_data$diabetes,
                              levels = c(0, 1),
                              labels = c("neg", "pos"))

# Split into training and testing sets
set.seed(123)
train_idx <- createDataPartition(model_data$diabetes, p = 0.7, list = FALSE)
train_data <- model_data[train_idx, ]
test_data <- model_data[-train_idx, ]

x_train <- train_data[, selected_vars]
y_train <- train_data$diabetes
x_test <- test_data[, selected_vars]
y_test <- test_data$diabetes

# Set up training control with 10-fold cross-validation
ctrl <- trainControl(method = "cv", number = 10,
                     classProbs = TRUE,
                     summaryFunction = twoClassSummary)

# Train Decision Tree model using rpart
dt_model <- train(x = x_train, y = y_train,
                  method = "rpart",
                  trControl = ctrl,
                  metric = "ROC",
                  tuneLength = 10)

# Display optimal parameters
print(dt_model)

# Predict on the test set
dt_pred <- predict(dt_model, x_test)
dt_prob <- predict(dt_model, x_test,_
