# Install required packages
install.packages("caret")
install.packages("e1071")  # Dependency for 'caret'
install.packages("class")  # Base implementation of KNN

# Load libraries
library(caret)
library(pROC)
library(e1071)
library(class)

# Load dataset
data <- read.csv("HN22_ALL_103124_fin.csv")

# Variables selected by LASSO + domain-specific important variables
selected_vars <- c("incm5", "DI1_pr", "DI2_pr", "HE_HTG", "BO1", "BE3_92", 
                   "HE_DMfh3", "HE_wc", "HE_glu", "HE_HDL_st2")

# Create analysis dataset
model_data <- data[, c("diabetes", selected_vars)]

# Remove missing values (KNN does not allow missing data)
model_data <- na.omit(model_data)

# Convert dependent variable to factor with proper labels
model_data$diabetes <- factor(model_data$diabetes,
                              levels = c(0, 1),
                              labels = c("neg", "pos"))  # Must be character labels

# Split into training and test sets (e.g., 70:30 ratio)
set.seed(123)
train_idx <- createDataPartition(model_data$diabetes, p = 0.7, list = FALSE)
train_data <- model_data[train_idx, ]
test_data <- model_data[-train_idx, ]

# Separate X and y from training set
x_train <- train_data[, selected_vars]
y_train <- train_data$diabetes

# Separate X and y from test set
x_test <- test_data[, selected_vars]
y_test <- test_data$diabetes

# Normalize features (KNN is distance-based, so scaling is essential)
preproc <- preProcess(x_train, method = c("center", "scale"))
x_train_scaled <- predict(preproc, x_train)
x_test_scaled <- predict(preproc, x_test)

# Train KNN model and tune hyperparameter k using caret
ctrl <- trainControl(method = "cv", number = 10, classProbs = TRUE, summaryFunction = twoClassSummary)
knn_model <- train(x = x_train_scaled, y = y_train,
                   method = "knn",
                   trControl = ctrl,
                   metric = "ROC",
                   tuneLength = 10)  # Search across various k values

# Display best k
print(knn_model)

# Predict on test set
knn_pred <- predict(knn_model, x_test_scaled)
knn_prob <- predict(knn_model, x_test_scaled, type = "prob")

# Confusion matrix
confusionMatrix(knn_pred, y_test)

# Calculate AUC
roc_obj <- roc(response = y_test, predictor = knn_prob[, 2])
auc_value <- auc(roc_obj)
cat("AUC value:", auc_value, "\n")

# Plot ROC curve
plot(roc_obj, col = "blue", lwd = 2, main = "ROC Curve for KNN")
abline(a = 0, b = 1, lty = 2, col = "red")  # Diagonal reference line
legend("bottomright", legend = paste("AUC =", round(auc_value, 3)), col = "blue", lwd = 2)
