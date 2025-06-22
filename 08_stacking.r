# Install required packages
install.packages("caret")
install.packages("caretEnsemble")
install.packages("e1071")
install.packages("nnet")
install.packages("randomForest")
install.packages("xgboost")
install.packages("pROC")

# Load packages
library(caret)
library(caretEnsemble)
library(e1071)
library(nnet)
library(randomForest)
library(xgboost)
library(pROC)

# Load dataset
data <- read.csv("HN22_ALL_103124_fin.csv")

# Select variables
selected_vars <- c("incm5", "DI1_pr", "DI2_pr", "HE_HTG", "BO1", "BE3_92", 
                   "HE_DMfh3", "HE_wc", "HE_glu", "HE_HDL_st2")

# Construct modeling dataset
model_data <- data[, c("diabetes", selected_vars)]
model_data <- na.omit(model_data)
model_data$diabetes <- factor(model_data$diabetes, levels = c(0, 1), labels = c("neg", "pos"))

# Split into training and test sets
set.seed(123)
train_idx <- createDataPartition(model_data$diabetes, p = 0.7, list = FALSE)
train_data <- model_data[train_idx, ]
test_data <- model_data[-train_idx, ]

# Apply scaling for consistency
preproc <- preProcess(train_data[, selected_vars], method = c("center", "scale"))
train_scaled <- predict(preproc, train_data[, selected_vars])
test_scaled <- predict(preproc, test_data[, selected_vars])
train_scaled$diabetes <- train_data$diabetes
test_scaled$diabetes <- test_data$diabetes

# Define training control
ctrl <- trainControl(method = "cv", number = 5,
                     savePredictions = "final",
                     classProbs = TRUE,
                     summaryFunction = twoClassSummary)

# Train base learners
set.seed(123)
models <- caretList(
  diabetes ~ ., 
  data = train_scaled,
  trControl = ctrl,
  metric = "ROC",
  tuneList = list(
    knn = caretModelSpec(method = "knn", tuneLength = 5),
    svmRadial = caretModelSpec(method = "svmRadial", tuneLength = 5),
    nnet = caretModelSpec(method = "nnet", tuneLength = 5, trace = FALSE),
    glm = caretModelSpec(method = "glm"),
    rpart = caretModelSpec(method = "rpart", tuneLength = 5),
    rf = caretModelSpec(method = "rf", tuneLength = 5),
    xgbTree = caretModelSpec(method = "xgbTree", tuneLength = 5)
  )
)

# Train stacked ensemble model (meta-learner = logistic regression)
stack_ctrl <- trainControl(method = "cv", number = 5,
                           savePredictions = "final",
                           classProbs = TRUE,
                           summaryFunction = twoClassSummary)

set.seed(123)
stack_model <- caretStack(models, method = "glm", metric = "ROC", trControl = stack_ctrl)

# Print model performance
print(stack_model)

# Predict on the test set
stack_pred <- predict(stack_model, newdata = test_scaled, type = "raw")
stack_prob <- predict(stack_model, newdata = test_scaled, type = "prob")

# Generate confusion matrix
confusion <- confusionMatrix(stack_pred, test_scaled$diabetes)
print(confusion)

# Compute ROC and AUC
roc_obj <- roc(response = test_scaled$diabetes, predictor = stack_prob[,"pos"])
auc_value <- auc(roc_obj)
cat("Stacking model AUC:", auc_value, "\n")

# Plot ROC curve
plot(roc_obj, col = "blue", lwd = 2, main = "ROC Curve for Stacked Model")
abline(a = 0, b = 1, lty = 2, col = "red")
legend("bottomright", legend = paste("AUC =", round(auc_value, 3)), col = "blue", lwd = 2)
