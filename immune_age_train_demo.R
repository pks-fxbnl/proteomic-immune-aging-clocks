
############################################
## Demo code for proteomic immune aging models
## - Data generation (train & test)
## - Immune age model (139 proteins)
############################################

### ----------------------------------------
### 0. Load required packages
### ----------------------------------------
library(glmnet)

### ----------------------------------------
### 1. Generate demo proteomics datasets
### ----------------------------------------

set.seed(123)  # for reproducibility

# 139 immune-related proteomic features
protein_names <- c(
  "CCL3", "KLRB1", "ICAM3", "VCAM1", "ITGAM",
  "RNASET2", "NINJ1", "SIGLEC7", "SLAMF6", "PDCD1",
  "SIGLEC9", "CD300E", "TNFRSF13C", "TNFRSF10C", "TREML2",
  "CD244", "CD38", "KLRD1", "SLAMF7", "CTSS",
  "PILRA", "CEACAM8", "TNFRSF13B", "CD27", "BTN3A2",
  "LY96", "GZMH", "LILRB2", "TNFRSF1B", "IL12RB1",
  "FCRL2", "GRAP2", "SIGLEC10", "CLEC4D", "CCL24",
  "FYB1", "CXCL6", "KIR2DL3", "AZU1", "FCER2",
  "GZMA", "CEACAM21", "APBB1IP", "CSF2RA", "LAT2",
  "FMNL1", "KIR3DL1", "DPEP2", "PTPN6", "DOK2",
  "CST7", "MPO", "CCL14", "SIRPB1", "CLC",
  "RELT", "XCL1", "LILRA5", "LILRA2", "FCRL5",
  "CD79B", "CD300LF", "AIF1", "LYN", "TNF",
  "CD33", "SIGLEC5", "PRTN3", "C1QA", "BANK1",
  "MMP9", "HMOX1", "TCL1A", "MMP8", "FOLR3",
  "CLEC6A", "S100A12", "CPPED1", "RNASE3", "FCRL6",
  "HCLS1", "PGLYRP1", "CORO1A", "IL1B", "OSM",
  "GZMB", "FCGR2A", "IL18RAP", "NCF2", "VSIR",
  "CCL4", "MZB1", "IL10RB", "CD177", "HBQ1",
  "AHSP", "CD48", "CCL5", "IFNGR2", "LILRB5",
  "ITGB7", "CD300C", "VSTM1", "CXCL1", "FCAR",
  "NOS3", "CLEC4A", "FCRL3", "CD160", "ADA2",
  "FGR", "FLI1", "GNLY", "CXCL13", "FCGR3B",
  "FCRL1", "MNDA", "SH2D1A", "LY9", "IL10RA",
  "NFKBIE", "LSP1", "CD6", "RETN", "IL17RA",
  "NCR1", "CR2", "CD8A", "OSCAR", "TNFRSF9",
  "LRRC25", "CD84", "LILRB1", "CD28", "KAZALD1",
  "BST1", "IL1R2", "CD5", "FASLG"
)

# Number of proteins
n_proteins <- length(protein_names)

# Verify feature list
stopifnot(n_proteins == 139)
stopifnot(!anyDuplicated(protein_names))

### 1.1 Training dataset: 98 healthy participants
n_train <- 98

train_demo <- data.frame(
  SampleID = paste0("Healthy_", seq_len(n_train)),
  matrix(
    rnorm(n_train * n_proteins, mean = 0, sd = 1),
    nrow = n_train,
    ncol = n_proteins
  )
)

colnames(train_demo)[-1] <- protein_names

### 1.2 Test dataset: 1850 HIV participants
n_test <- 1850

test_demo <- data.frame(
  SampleID = paste0("HIV_", seq_len(n_test)),
  matrix(
    rnorm(n_test * n_proteins, mean = 0, sd = 1),
    nrow = n_test,
    ncol = n_proteins
  )
)

colnames(test_demo)[-1] <- protein_names

### 1.3 Convert SampleID to rownames and drop column

# Training dataset
rownames(train_demo) <- train_demo$SampleID
train_demo$SampleID <- NULL

# Test dataset
rownames(test_demo) <- test_demo$SampleID
test_demo$SampleID <- NULL

# Quick checks (optional)
# dim(train_demo)  # 98 × 139
# dim(test_demo)   # 1850 × 139
# head(train_demo[, 1:5])

### 1.4 Generate demo chronological ages

# Train: 98 healthy participants (age 20–80)
set.seed(123)
train_age_demo <- round(runif(n_train, min = 20, max = 80))

# Test: 1850 HIV participants
set.seed(456)
test_age_demo <- round(runif(n_test, min = 20, max = 80))

# Quick checks (optional)
# length(train_age_demo)  # 98
# length(test_age_demo)   # 1850



### ----------------------------------------
### 2. Helper function: lambda-ensemble LASSO
### ----------------------------------------

lambda_ensemble_lasso <- function(x_train, y_train, x_test,
                                  n_bootstrap = 500,
                                  seed = 1234) {
  set.seed(seed)
  
  n_train <- nrow(x_train)
  n_test  <- nrow(x_test)
  
  # Matrices to store predictions from each bootstrap model
  train_pred_mat <- matrix(NA_real_, nrow = n_train, ncol = n_bootstrap)
  test_pred_mat  <- matrix(NA_real_, nrow = n_test,  ncol = n_bootstrap)
  
  # Vector to store lambda from each bootstrap iteration
  lambda_vec <- numeric(n_bootstrap)
  
  # Data frame to store loss metrics per iteration
  loss_table <- data.frame(
    Iteration = integer(n_bootstrap),
    Lambda    = numeric(n_bootstrap),
    CV_Error  = numeric(n_bootstrap),
    R2        = numeric(n_bootstrap)
  )
  
  # List to store coefficients (including intercept)
  # from each bootstrap model
  coef_list <- vector("list", n_bootstrap)
  
  for (b in seq_len(n_bootstrap)) {
    
    # 1. Bootstrap resampling of the training set
    sample_idx <- sample.int(n_train, size = n_train, replace = TRUE)
    X_boot <- x_train[sample_idx, , drop = FALSE]
    y_boot <- y_train[sample_idx]
    
    # 2. Cross-validated LASSO on bootstrap sample
    #    to select optimal lambda
    cv_fit <- cv.glmnet(
      X_boot, y_boot,
      alpha        = 1,
      nfolds       = 5,
      type.measure = "mse"
    )
    
    lambda_opt <- cv_fit$lambda.min
    lambda_vec[b] <- lambda_opt
    
    # 3. Fit LASSO model on bootstrap sample
    #    with selected lambda
    fit_boot <- glmnet(
      X_boot, y_boot,
      alpha  = 1,
      lambda = lambda_opt
    )
    
    # 4. Store coefficients (including intercept)
    coef_vec <- as.vector(coef(fit_boot))
    names(coef_vec) <- rownames(coef(fit_boot))
    
    coef_list[[b]] <- data.frame(
      Feature     = names(coef_vec),
      Coefficient = coef_vec,
      Iteration   = b,
      row.names   = NULL
    )
    
    # 5. Compute R² on bootstrap sample
    y_boot_pred <- as.numeric(predict(fit_boot, X_boot))
    
    r2_boot <- 1 - sum((y_boot - y_boot_pred)^2) /
      sum((y_boot - mean(y_boot))^2)
    
    # 6. Predict immune age on full training data
    #    using this bootstrap model
    train_pred_mat[, b] <- as.numeric(
      predict(fit_boot, x_train)
    )
    
    # 7. Refit on full training data
    #    using the same selected lambda
    fit_full <- glmnet(
      x_train, y_train,
      alpha  = 1,
      lambda = lambda_opt
    )
    
    # 8. Predict immune age on the HIV test dataset
    test_pred_mat[, b] <- as.numeric(
      predict(fit_full, x_test)
    )
    
    # 9. Record model performance metrics
    loss_table$Iteration[b] <- b
    loss_table$Lambda[b]    <- lambda_opt
    loss_table$CV_Error[b]  <- min(cv_fit$cvm, na.rm = TRUE)
    loss_table$R2[b]        <- r2_boot
  }
  
  # Ensemble predictions: mean across 500 bootstrap models
  train_pred_mean <- rowMeans(train_pred_mat, na.rm = TRUE)
  test_pred_mean  <- rowMeans(test_pred_mat,  na.rm = TRUE)
  
  list(
    train_pred     = train_pred_mean,
    test_pred      = test_pred_mean,
    lambda_vec     = lambda_vec,
    loss_table     = loss_table,
    coefficients   = coef_list,
    train_pred_mat = train_pred_mat,
    test_pred_mat  = test_pred_mat
  )
}


### ----------------------------------------
### 3. Proteomic IMM-age model (139 immune proteins)
### ----------------------------------------

# Convert demo proteomic datasets to matrices
# Both datasets contain the same 139 immune-related proteins

x_train_immune <- as.matrix(train_demo)  # 98 × 139
x_test_immune  <- as.matrix(test_demo)   # 1850 × 139

# Chronological age of healthy training participants
y_train <- as.numeric(train_age_demo)

# Verify input dimensions and feature consistency
stopifnot(ncol(x_train_immune) == 139)
stopifnot(identical(
  colnames(x_train_immune),
  colnames(x_test_immune)
))

# Train the proteomic IMM-age model and
# predict immune age in the HIV test cohort
immune_res <- lambda_ensemble_lasso(
  x_train     = x_train_immune,
  y_train     = y_train,
  x_test      = x_test_immune,
  n_bootstrap = 500,
  seed        = 2025
)

# Predicted proteomic immune age in the HIV cohort
immune_age_demo <- immune_res$test_pred

# Predicted proteomic immune age in the training cohort
immune_age_train_demo <- immune_res$train_pred

# Optional checks
# summary(immune_age_demo)
# head(immune_age_demo)
# head(immune_res$loss_table)
# head(immune_res$coefficients[[1]])



###########################################
## End of IMM-age demo script
###########################################


