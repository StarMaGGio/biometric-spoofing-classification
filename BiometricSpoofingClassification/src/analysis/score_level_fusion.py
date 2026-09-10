from src.models.gaussian_mixture_models import GaussianMixtureModel
from src.models.logistic_regression import WeightedLogisticRegression
from src.models.support_vector_machines import KernelSupportVectorMachine
from src.models.utils import quadratic_expansion, rbfKernel, split_db_2to1
from src.models.visualization import plot_Bayes_error

import numpy as np

import math

def score_level_fusion(D, L):

    # ---------------------
    #  Scores Level Fusion
    # ---------------------

    # Divide the dataset in training and validation sets
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)

    print("\nTraining models for score fusion")

    # Train three different models (Logistic Regression, Support Vector Machine, Gaussian Mixture Model)
    # NOTE: quadratic expansion is applied only for WLR; SVM and GMM use the original features,
    # consistent with how each model was individually tuned in scores_calibration.py.
    DTR_quad = quadratic_expansion(DTR)
    DVAL_quad = quadratic_expansion(DVAL)

    lamb = 10 ** -3
    WLR = WeightedLogisticRegression()
    WLR.train(DTR_quad, LTR, lamb)
    raw_scores_lr = WLR.get_log_likelihood_ratios(DVAL_quad) # Raw validation scores for Weighted Logistic Regression

    kernelFunc = rbfKernel(math.exp(-1))
    C = 1
    KSVM = KernelSupportVectorMachine()
    KSVM.train(DTR, LTR, C, kernelFunc)
    raw_scores_svm = KSVM.get_scores(DVAL) # Raw validation scores for SVM with RBF Kernel

    num_components = 16
    alpha = 0.1
    psi = 0.01
    GMM = GaussianMixtureModel()
    GMM.train(DTR, LTR, numComponents=num_components, alpha=alpha, psi=psi)
    raw_scores_gmm = GMM.get_scores(DVAL) # Raw validation scores for GMM with 16 components

    print("\n Computing score-level fusion of the three models")

    # Compute score-level fusion of the three models (weighted logistic regression, SVM with RBF kernel, GMM with 8 components)
    # Make a matrix of shape (3, N) where N is the number of samples in the validation set, each row represents the scores of a model
    raw_scores_fusion = np.vstack([raw_scores_lr, raw_scores_svm, raw_scores_gmm])

    # TODO: Make this a function and move in "cross_validation.py"
    # Apply k fold cross-validation to train the fusion model (logistic regression) on the validation set with the application prior (pEmp)
    K = 5
    raw_scores_fusion_folds = [raw_scores_fusion[:, i::K] for i in range(K)]
    LVAL_folds = [LVAL[i::K] for i in range(K)]

    calibrated_scores_fusion = np.zeros_like(raw_scores_fusion[0])
    for k in range(K):
        # Train the calibration model on K-1 folds and validate on the remaining fold
        SCAL, SVAL_k = np.hstack([raw_scores_fusion_folds[i] for i in range(K) if i != k]), raw_scores_fusion_folds[k]
        LCAL, LVAL_k = np.hstack([LVAL_folds[i] for i in range(K) if i != k]), LVAL_folds[k]

        # Train calibration model (weighted logistic regression) on the calibration training set with the application prior (piT=0.1)
        lamb = 1e-3
        WLR_cal = WeightedLogisticRegression()
        WLR_cal.train(SCAL, LCAL, lamb)

        # Compute calibrated scores on the validation fold
        calibrated_scores_fusion[k::K] = WLR_cal.get_log_likelihood_ratios(SVAL_k)

    print("\n Computing DCFs on calibrated scores for the fused system")
    # Compute and print DCF for the fused system
    plot_Bayes_error(calibrated_scores_fusion, LVAL, "Fused System (Weighted Logistic Regression)")