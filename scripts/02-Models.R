################################################################################
# Install and run packages, set seed and import data.                          #
################################################################################

install.packages("caret")
install.packages("pROC")
install.packages("dplyr")
install.packages("randomForest")
install.packages("xgboost")
install.packages("keras3")

library(caret)
library(pROC)
library(dplyr)
library(randomForest)
library(xgboost)
library(keras3)

set.seed(123)

dir.create("results", showWarnings = FALSE)

proc_data <- readRDS("preprocessed_data.rds")
x_train <- proc_data$x_train; y_train <- proc_data$y_train
x_test  <- proc_data$x_test;  y_test  <- proc_data$y_test

################################################################################
# Create function to compare results.                                          #
################################################################################

eval_binary <- function(y_true, prob, threshold = 0.5) {
  stopifnot(length(prob) == length(y_true))
  y_true <- factor(y_true, levels = c("N", "Y"))
  
  pred <- ifelse(prob >= threshold, "Y", "N") |> factor(levels = c("N", "Y"))
  
  cm <- confusionMatrix(pred, y_true, positive = "Y")
  
  roc_obj <- pROC::roc(response = y_true, predictor = prob, levels = c("N", "Y"), direction = "<")
  auc <- as.numeric(pROC::auc(roc_obj))
  
  out <- list(
    threshold = threshold,
    auc = auc,
    accuracy = cm$overall["Accuracy"],
    precision = cm$byClass["Precision"],
    recall = cm$byClass["Recall"],
    f1 = cm$byClass["F1"],
    confusion = cm$table
  )
  return(out)
}

################################################################################
# Convert ot matrices for xgboost and keras                                    #
################################################################################

x_train_mat <- as.matrix(x_train)
x_test_mat  <- as.matrix(x_test)

################################################################################
# 1) Logistic Regression                                                       #
################################################################################

glm_train <- cbind(fraud_reported = y_train, x_train)
glm_fit <- glm(fraud_reported ~ ., data = glm_train, family = binomial())
glm_prob <- predict(glm_fit, newdata = x_test, type = "response")
glm_metrics <- eval_binary(y_test, glm_prob, threshold = 0.5)

saveRDS(glm_fit, "results/model_glm.rds")
saveRDS(glm_metrics, "results/metrics_glm.rds")

################################################################################
# 2) Random Forest                                                             #
################################################################################

rf_train <- cbind(fraud_reported = y_train, x_train)
rf_fit <- randomForest(fraud_reported ~ ., data = rf_train, ntree = 500, mtry = floor(sqrt(ncol(x_train))))
rf_prob <- predict(rf_fit, newdata = x_test, type = "prob")[, "Y"]
rf_metrics <- eval_binary(y_test, rf_prob, threshold = 0.5)

saveRDS(rf_fit, "results/model_rf.rds")
saveRDS(rf_metrics, "results/metrics_rf.rds")

################################################################################
# 3) Gradient Boosting                                                         #
################################################################################

y_train_bin <- ifelse(y_train == "Y", 1, 0)
y_test_bin  <- ifelse(y_test == "Y", 1, 0)
dtrain <- xgb.DMatrix(data = x_train_mat, label = y_train_bin)
dtest  <- xgb.DMatrix(data = x_test_mat,  label = y_test_bin)

pos <- sum(y_train_bin == 1)
neg <- sum(y_train_bin == 0)
scale_pos_weight <- ifelse(pos == 0, 1, neg / pos)

params <- list(
  objective = "binary:logistic",
  eval_metric = "auc",
  eta = 0.05,
  max_depth = 4,
  subsample = 0.8,
  colsample_bytree = 0.8,
  scale_pos_weight = scale_pos_weight
)

xgb_fit <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 500,
  watchlist = list(train = dtrain),
  verbose = 0
)

xgb_prob <- predict(xgb_fit, dtest)
xgb_metrics <- eval_binary(y_test, xgb_prob, threshold = 0.5)

xgb.save(xgb_fit, "results/model_xgb.json")
saveRDS(xgb_metrics, "results/metrics_xgb.rds")

################################################################################
# 4) Neural Network                                                            #
################################################################################

keras::k_clear_session()

y_train_nn <- ifelse(y_train == "Y", 1, 0)
y_test_nn  <- ifelse(y_test == "Y", 1, 0)

nn_model <- keras_model_sequential() |>
  layer_dense(units = 64, activation = "relu", input_shape = ncol(x_train_mat)) |>
  layer_dropout(rate = 0.3) |>
  layer_dense(units = 32, activation = "relu") |>
  layer_dropout(rate = 0.3) |>
  layer_dense(units = 1, activation = "sigmoid")

nn_model |>
  compile(
    optimizer = optimizer_adam(learning_rate = 1e-3),
    loss = "binary_crossentropy",
    metrics = c("accuracy")
  )

history <- nn_model |>
  fit(
    x = x_train_mat,
    y = y_train_nn,
    epochs = 50,
    batch_size = 32,
    validation_split = 0.2,
    callbacks = list(callback_early_stopping(patience = 6, restore_best_weights = TRUE)),
    verbose = 0
  )

nn_prob <- as.numeric(nn_model |> predict(x_test_mat, verbose = 0))
nn_metrics <- eval_binary(y_test, nn_prob, threshold = 0.5)

save_model(nn_model, "results/model_nn.keras")
saveRDS(history, "results/history_nn.rds")
saveRDS(nn_metrics, "results/metrics_nn.rds")

################################################################################
# Create Summary Table                                                         #
################################################################################

summary_tbl <- tibble::tibble(
  model = c("GLM (Logistic)", "Random Forest", "XGBoost", "Neural Net"),
  auc = c(glm_metrics$auc, rf_metrics$auc, xgb_metrics$auc, nn_metrics$auc),
  accuracy = c(glm_metrics$accuracy, rf_metrics$accuracy, xgb_metrics$accuracy, nn_metrics$accuracy),
  precision = c(glm_metrics$precision, rf_metrics$precision, xgb_metrics$precision, nn_metrics$precision),
  recall = c(glm_metrics$recall, rf_metrics$recall, xgb_metrics$recall, nn_metrics$recall),
  f1 = c(glm_metrics$f1, rf_metrics$f1, xgb_metrics$f1, nn_metrics$f1)
)

write.csv(summary_tbl, "results/metrics_summary.csv", row.names = FALSE)
print(summary_tbl)
