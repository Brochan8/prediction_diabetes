# Load required packages
library(caret)
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

# Train logistic regression model using glm
logit_model <- glm(diabetes ~ ., data = train_data, family = binomial)

# Display model summary
summary(logit_model)

# Generate predicted probabilities on the test set
logit_prob <- predict(logit_model, newdata = test_data, type = "response")

# Convert probabilities to binary class predictions (threshold = 0.5)
logit_pred <- ifelse(logit_prob > 0.5, "pos", "neg")
logit_pred <- factor(logit_pred, levels = c("neg", "pos"))

# Actual test labels
y_test <- test_data$diabetes

# Confusion matrix
confusionMatrix(logit_pred, y_test)

# ROC and AUC calculation
roc_obj <- roc(response = y_test, predictor = logit_prob)
auc_value <- auc(roc_obj)
cat("AUC value:", auc_value, "\n")

# Plot ROC curve
plot(roc_obj, col = "blue", lwd = 2, main = "ROC Curve for Logistic Regression")
abline(a = 0, b = 1, lty = 2, col = "red")  # Diagonal reference line
legend("bottomright", legend = paste("AUC =", round(auc_value, 3)), col = "blue", lwd = 2)
