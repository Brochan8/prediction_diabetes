# Install required package
install.packages("randomForest")

# Load packages
library(caret)
library(randomForest)
library(pROC)

# Variables selected from LASSO + manually added variables
selected_vars <- c("incm5", "DI1_pr", "DI2_pr", "HE_HTG", "BO1", "BE3_92", 
                   "HE_DMfh3", "HE_wc", "HE_glu", "HE_HDL_st2")

# Create modeling dataset
model_data <- data[, c("diabetes", selected_vars)]

# Remove missing values
model_data <- na.omit(model_data)

# Convert target variable to factor
model_data$diabetes <- factor(model_data$diabetes,
                              levels = c(0, 1),
                              labels = c("neg", "pos"))

# Split into training and testing sets
set.seed(123)
train_idx <- createDataPartition(model_data$diabetes, p = 0.7, list = FALSE)
train_data <- model_data[train_idx, ]
test_data <- model_data[-train_idx, ]

# Separate features and labels
x_train <- train_data[, selected_vars]
y_train <- train_data$diabetes
x_test <- test_data[, selected_vars]
y_test <- test_data$diabetes

# Set up training control with 10-fold cross-validation
ctrl <- trainControl(method = "cv", number = 10,
                     classProbs = TRUE,
                     summaryFunction = twoClassSummary)

# Train Random Forest model (may take time)
rf_model <- train(x = x_train, y = y_train,
                  method = "rf",
                  trControl = ctrl,
                  metric = "ROC",
                  tuneLength = 5)  # Tune using various mtry values

# Display optimal parameters
print(rf_model)

# Predict on the test set
rf_pred <- predict(rf_model, x_test)
rf_prob <- predict(rf_model, x_test, type = "prob")

# Print confusion matrix
confusionMatrix(rf_pred, y_test)

# Compute ROC and AUC
roc_obj <- roc(response = y_test, predictor = rf_prob[,"pos"])
auc_value <- auc(roc_obj)
cat("AUC value:", auc_value, "\n")

# Plot ROC curve
plot(roc_obj, col = "blue", lwd = 2, main = "ROC Curve for Random Forest")
abline(a = 0, b = 1, lty = 2, col = "red")  # Diagonal reference line
legend("bottomright", legend = paste("AUC =", round(auc_value, 3)), col = "blue", lwd = 2)

# Visualize variable importance
varImpPlot(rf_model$finalModel, main = "Variable Importance (RF)")
