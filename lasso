setwd("C:/Users/LG/Desktop/na/prediction_KHANES/KHANES/code")

# 필요한 패키지 설치
install.packages("glmnet")
install.packages("pROC")

# 패키지 로드
library(glmnet)
library(pROC)


# 데이터 불러오기 (CSV 파일을 예시로 사용)
data <- read.csv("HN22_ALL_103124_fin.csv") # 데이터 파일 경로를 입력하세요.


# 종속변수 (diabetes)와 독립변수 설정
# 독립변수는 범주형 변수를 One-Hot Encoding하여 처리
X <- model.matrix(diabetes ~ sex + nwo + incm5 + cfam + marri_1 + npins + educ + EC1_1 
                  + DI1_pr + DI2_pr + DJ4_3 + BP17_dg + HE_HCHOL + HE_HTG + HE_obe + BO1 
                  + OR1 + BP1 + BP7 + BO1_3 + BO2_1 + BD1_11 + BD2_1 + BS3_1 + BS3_2 
                  + L_BR_FQ + L_LN_FQ + L_DN_FQ + L_OUT_FQ + LK_LB_US + BE3_92 + BE3_76
                  + BE3_86 + BE8_1 + BE3_31 + BE3_32 + BE5_1 + pa_aerobic + HE_fh 
                  + HE_DMfh1 + HE_DMfh2 + HE_DMfh3 
                  + HE_wc + HE_glu + HE_HDL_st2 + HE_TG + astalt 
                  + N_INTK + N_SFA + N_CHOL + N_SUGAR + N_NA - 1, data) # 모든 변수 포함, 상수항 제거
y <- data$diabetes # 종속변수


# LASSO 로지스틱 회귀 수행
lasso_model <- cv.glmnet(X, y, family = "binomial", alpha = 1, nfolds = 10)


# 최적의 람다 값 확인
best_lambda <- lasso_model$lambda.min
cat("최적의 람다 값:", best_lambda, "\n")


# 선택된 변수 확인
selected_coefs <- coef(lasso_model, s = best_lambda) # 최적 람다에서의 계수
selected_vars <- rownames(selected_coefs)[selected_coefs[, 1] != 0]
cat("LASSO에 의해 선택된 변수들:\n", selected_vars, "\n")


# 연구에서 중요한 변수 추가
important_vars <- c("HE_HDL_st2", "HE_glu", "HE_wc") # 연구적으로 중요한 변수들
final_vars <- union(selected_vars, important_vars) # LASSO 선택 변수와 중요한 변수 결합


# 예측값 생성 (훈련 데이터로 예측)
predicted_probabilities <- predict(lasso_model, newx = X, s = best_lambda, type = "response")


# 임계값을 사용하여 클래스 예측 (0.5를 기준으로 설정)
predicted_classes <- ifelse(predicted_probabilities > 0.5, 1, 0)


# 예측 결과 확인
table(Predicted = predicted_classes, Actual = y)


# AUC 계산
roc_obj <- roc(y, as.numeric(predicted_probabilities))
auc(roc_obj)
auc_value <- auc(roc_obj)
cat("AUC 값:", auc_value, "\n")


# ROC 곡선 그리기
plot(roc_obj, col = "blue", lwd = 2, main = "ROC Curve for LASSO Logistic Regression")
abline(a = 0, b = 1, lty = 2, col = "red") # 대각선 참고선
legend("bottomright", legend = paste("AUC =", round(auc_value, 3)), col = "blue", lwd = 2)

