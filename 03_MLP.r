# Install required package
install.packages("nnet")

# Load packages
library(nnet)
library(pROC)
library(caret)

# Variables selected from LASSO + manually added important variables
selected_vars <- c("incm5", "DI1_pr", "DI2_pr", "HE_HTG", "BO1", "BE3_92", 
                   "HE_DMfh3", "HE_wc", "HE_glu", "HE_HDL_st2")

# Create modeling dataset
model_data <- data[, c("diabetes", selected_vars)]

# Remove missing values
model_data <- na.omit(model_data)

# Convert the dependent variable to a factor (caret-friendly format)
model_data$diabetes <- factor(model_data$diabetes,
                              levels = c(0, 1),
                              labels = c("neg", "pos"))

# Split data into training and testing sets
set.seed(123)
train_idx <- createDataPartition(model_data$diabetes, p = 0.7, list = FALSE)
train_data <- model_data[train_idx, ]
test_data <- model_data[-train_idx, ]

# Separate predictors and response variables
x_train <- train_data[, selected_vars]
y_train <- train_data$diabetes
x_test <- test_data[, selected_vars]
y_test <- test_data$diabetes

# Normalize features (MLP is sensitive to scale)
preproc <- preProcess(x_train, method = c("center", "scale"))
x_train_scaled <- predict(preproc, x_train)
x_test_scaled <- predict(preproc, x_test)

# Set up training control with 10-fold cross-validation
ctrl <- trainControl(method = "cv", number = 10,
                     classProbs = TRUE,
                     summaryFunction = twoClassSummary)

# Train MLP model (may take some time)
mlp_model <- train(x = x_train_scaled, y = y_train,
                   method = "nnet",
                   trControl = ctrl,
                   metric = "ROC",
                   tuneLength = 10,
                   trace = FALSE)  # Suppress console output

# Print best model info
print(mlp_model)

# Predict on the test set
mlp_pred <- predict(mlp_model, x_test_scaled)
mlp_prob <- predict(mlp_model, x_test_scaled, type = "prob")

# Confusion matrix
confusionMatrix(mlp_pred, y_test)

# Calculate ROC and AUC
roc_obj <- roc(response = y_test, predictor = mlp_prob[,"pos"])
auc_value <- auc(roc_obj)
cat("AUC value:", auc_value, "\n")

# Plot ROC curve
plot(roc_obj, col = "blue", lwd = 2, main = "ROC Curve for MLP")
abline(a = 0, b = 1, lty = 2, col = "red")
legend("bottomright", legend = paste("AUC =", round(auc_value, 3)), col = "blue", lwd = 2)
