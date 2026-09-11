# --------------------
#  Evaluation Dataset
# --------------------
from src.models.gaussian_mixture_models import GaussianMixtureModel
from src.models.logistic_regression import WeightedLogisticRegression
from src.models.support_vector_machines import KernelSupportVectorMachine
from src.models.utils import rbfKernel
from src.models.visualization import plot_Bayes_error

import numpy as np

import math

def final_evaluation(DTR, LTR, DEVAL, LEVAL):
    # Hyperparameters for the models (already optimized in previous analysis)
    lamb_lr = 1e-3  # Regularization parameter for Weighted Logistic Regression
    C_svm = 1 # Regularization parameter for SVM
    kernelFunc = rbfKernel(math.exp(-1)) # RBF kernel with gamma = exp(-1)
    num_components_gmm = 8 # Number of components for GMM
    alpha_gmm = 0.1 # Alpha parameter for GMM
    psi_gmm = 0.01 # Psi parameter for GMM

    N_train = DTR.shape[1]

    # Initialize array to store Out-of-Fold raw scores for the full DTR dataset
    oof_scores_LR = np.zeros(N_train)
    oof_scores_SVM = np.zeros(N_train)
    oof_scores_GMM = np.zeros(N_train)

    # =================================================================================
    # --- PHASE 1: K-Fold on DTR to obtain Out-of-Fold Raw Scores (OOF) ---
    # =================================================================================
    K = 5
    print(f"Performing {K}-Fold Cross-Validation on DTR to obtain OOF scores (Phase 1)...")

    for k in range(K):
        # Indexes for validation and training folds
        idx_val = list(range(k, N_train, K)) # Extract every K-th index starting from k for validation
        idx_train = [i for i in range(N_train) if i not in idx_val]

        D_fold_train, L_fold_train = DTR[:, idx_train], LTR[idx_train]
        D_fold_val = DTR[:, idx_val]

        # 1. Train and predict with WLR
        WLR = WeightedLogisticRegression()
        WLR.train(D_fold_train, L_fold_train, lamb_lr)
        oof_scores_LR[idx_val] = WLR.get_log_likelihood_ratios(D_fold_val)

        # 2. Train and predict with KSVM
        KSVM = KernelSupportVectorMachine()
        KSVM.train(D_fold_train, L_fold_train, C_svm, kernelFunc)
        oof_scores_SVM[idx_val] = KSVM.get_scores(D_fold_val)

        # 3. Train and predict with GMM
        GMM = GaussianMixtureModel()
        GMM.train(D_fold_train, L_fold_train, num_components_gmm, alpha_gmm, psi_gmm)
        oof_scores_GMM[idx_val] = GMM.get_scores(D_fold_val)

    # Join OOF raw scores in a single matrix (Features: 3, Samples: N_train)
    S_matrix_TR = np.vstack((oof_scores_LR, oof_scores_SVM, oof_scores_GMM))

    # =================================================================================
    # --- PHASE 2: K-Fold to Evaluate Scores Level Fusion Effectiveness (optional) ---
    # Apply a K_Fold cross validation on 'S_matrix_TR' to evaluate the effectiveness
    # of score-level fusion and calibration. (Already done in previous analysis)
    # =================================================================================
    print("Phase 2: SKIPPED")

    # =================================================================================
    # --- PHASE 3: Prepare the final system (Base Models and Fuser/Calibrator) ---
    # =================================================================================
    print("Training Final System on the 100% DTR dataset (Phase 3)...")

    # 1. Train the Three Models on the ENTIRE DTR dataset
    Final_WLR = WeightedLogisticRegression()
    Final_WLR.train(DTR, LTR, lamb_lr)

    Final_KSVM = KernelSupportVectorMachine()
    Final_KSVM.train(DTR, LTR, C_svm, kernelFunc)

    Final_GMM = GaussianMixtureModel()
    Final_GMM.train(DTR, LTR, num_components_gmm, alpha_gmm, psi_gmm)

    # 2. Train the Fusion/Calibration model on the OOF scores obtained in Phase 1
    Fuser = WeightedLogisticRegression()
    Fuser.train(S_matrix_TR, LTR, lamb_lr)

    # =================================================================================
    # --- PHASE 4: Final evaluation on DEVAL ---
    # =================================================================================
    print("Inference and Evaluation on DEVAL dataset (Phase 4)...")

    # 1. Obtain raw scores from the three base models on DEVAL
    eval_scores_LR = Final_WLR.get_log_likelihood_ratios(DEVAL)
    eval_scores_SVM = Final_KSVM.get_scores(DEVAL)
    eval_scores_GMM = Final_GMM.get_scores(DEVAL)

    # 2. Join the raw scores in a single matrix (Features: 3, Samples: N_eval)
    S_matrix_EVAL = np.vstack((eval_scores_LR, eval_scores_SVM, eval_scores_GMM))

    # 3. Obtain the final fused and calibrated scores using the trained fusion model
    final_fused_scores = Fuser.get_log_likelihood_ratios(S_matrix_EVAL)

    # 4. Plot Bayes Errors over different applications
    plot_Bayes_error(final_fused_scores, LEVAL, "Final System Evaluation on DEVAL")