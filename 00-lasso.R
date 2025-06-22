setwd("C:/Users/LG/Desktop/na/prediction_KHANES/KHANES/code")

# Install required packages
install.packages("glmnet")
install.packages("pROC")

# Load packages
library(glmnet)
library(pROC)

# Load data (using a CSV file as an example)
data <- read.csv("HN22_ALL_103124_fin.csv") # Specify the correct path to your data file

# Define dependent variable (diabetes) and independent variables
# Apply One-Hot Encoding for categorical variables
X <- model.matrix(diabetes ~ sex + nwo + incm5 + cfam + marri_1 + npins + educ + EC1_1 
                  + DI1_pr + DI2_pr + DJ4_3 + BP17_dg + HE_HCHOL + HE_HTG + HE_obe + BO1 
                  + OR1 + BP1 + BP7 + BO1_3 + BO2_1 + BD1_11 + BD2_1 + BS3_1 + BS3_2 
                  + L_BR_FQ + L_LN_FQ + L_DN_FQ + L_OUT_FQ + LK_LB_US + BE3_92 + BE3_76
                  + BE3_86 + BE8_1 + BE3_31 + BE3_32 + BE5_1 + pa_aerobic + HE_fh 
                  + HE_DMfh1 + HE_DMfh2 + HE_DMfh3 
                  + HE_wc + HE_glu + HE_HDL_st2 + HE_TG + astalt 
                  + N_INTK + N_SFA + N_CHOL + N_SUGAR + N_NA - 1, data) # Include all variables, exclude intercept
y <- data$diabetes # Dependent variable

# Perform LASSO logistic regression
lasso_model <- cv.glmnet(X, y, family = "binomial", alpha = 1, nfolds = 10)

# Check optimal lambda value
best_lambda <- lasso_model$lambda.min
cat("Optimal lambda value:", best_lambda, "\n")

# Check selected variables
selected_coefs <- coef(lasso_model, s = best_lambda) # Coefficients at optimal lambda
selected_vars <- rownames(selected_coefs)[selected_coefs[, 1] != 0]
cat("Variables selected by LASSO:\n", selected_vars, "\n")

# Add research-prioritized variables
important_vars <- c("HE_HDL_st2", "HE_glu", "HE_wc") # Variables considered important in the research
final_vars <- union(selected_vars, important_vars) # Combine LASSO-selected and important variables

# Generate predicted probabilities (on training data)
predicted_probabilities <- predict(lasso_model, newx = X, s = best_lambda, type = "response")

# Predict classes using threshold (0.5)
predicted_classes <- ifelse(predicted_probabilities > 0.5, 1, 0)

# Check prediction results
table(Predicted = predicted_classes, Actual = y)

# Calculate AUC
roc_obj <- roc(y, as.numeric(predicted_probabilities))
auc(roc_obj)
auc_value <- auc(roc_obj)
cat("AUC value:", auc_value, "\n")

# Plot ROC curve
plot(roc_obj, col = "blue", lwd = 2, main = "ROC Curve for LASSO Logistic Regression")
abline(a = 0, b = 1, lty = 2, col = "red") # Diagonal reference line
legend("bottomright", legend = paste("AUC =", round(auc_value, 3)), col = "blue", lwd = 2)
