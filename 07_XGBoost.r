# Install required package
install.packages("xgboost")

# Load packages
library(caret)
library(xgboost)
library(pROC)

# Variables selected from LASSO + manually added important variables
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

x_train <- train_data[, selected_vars]
y_train <- train_data$diabetes
x_test <- test_data[, selected_vars]
y_test <- test_data$diabetes

# Feature scaling (XGBoost is usually robust to scaling, but done here for consistency)
preproc <- preProcess(x_train, method = c("center", "scale"))
x_train_scaled <- predict(preproc, x_train)
x_test_scaled <- predict(preproc, x_test)

# Set up training control with 5-fold cross-validation
ctrl <- trainControl(method = "cv", number = 5,
                     classProbs = TRUE,
                     summaryFunction = twoClassSummary)

# Train XGBoost model
xgb_model <- train(x = x_train_scaled, y = y_train,
                   method = "xgbTree",
                   trControl = ctrl,
                   metric = "ROC",
                   tuneLength = 5)

# Display optimal parameters
print(xgb_model)

# Predict on the test set
xgb_pred <- predict(xgb_model, x_test_scaled)
xgb_prob <- predict(xgb_model, x_test_scaled, type = "prob")

# Confusion matrix
confusionMatrix(xgb_pred, y_test)

# Compute ROC and AUC
roc_obj <- roc(response = y_test, predictor = xgb_prob[,"pos"])
auc_value <- auc(roc_obj)
cat("AUC value:", auc_value, "\n")

# Plot ROC curve
plot(roc_obj, col = "blue", lwd = 2, main = "ROC Curve for XGBoost")
abline(a = 0, b = 1, lty = 2, col = "red")  # Diagonal reference line
legend("bottomright", legend = paste("AUC =", round(auc_value, 3)), col = "blue", lwd = 2)

# Plot variable importance
xgb_imp <- varImp(xgb_model)
plot(xgb_imp, top = 10, main = "Variable Importance (XGBoost)")
