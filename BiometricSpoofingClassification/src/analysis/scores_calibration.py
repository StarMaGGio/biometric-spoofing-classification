from src.models.bayes_decisions_model import compute_actual_DCF, compute_minimum_DCF, compute_optimal_bayes_decisions
from src.models.gaussian_mixture_models import GaussianMixtureModel
from src.models.logistic_regression import WeightedLogisticRegression
from src.models.support_vector_machines import KernelSupportVectorMachine
from src.models.utils import compute_confusion_matrix, quadratic_expansion, rbfKernel, split_db_2to1, vrow
import matplotlib.pyplot as plt
import numpy as np
import math


def scores_calibration(D, L):

    # --------------------
    #  Scores Calibration
    # --------------------

    # Divide the dataset in training and validation sets
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)

    inner_menu_option = int(input('\n Choose a model to evaluate:\n'
                                  '1. Weighted Logistic Regression\n'
                                  '2. SVM RBF Kernel\n'
                                  '3. GMM\n'
                                  '0. Back\n'))
    model = ""
    raw_scores = None

    if inner_menu_option == 0: return

    K = int(input("Choose number of K-fold partitions: "))

    match inner_menu_option:
        case 1:
            DTR = quadratic_expansion(DTR)
            DVAL = quadratic_expansion(DVAL)
            model = "Weighted Logistic Regression Quadratic Expansion"
            lamb = 10 ** -3
            WLR = WeightedLogisticRegression()
            WLR.train(DTR, LTR, lamb)
            raw_scores = WLR.get_log_likelihood_ratios(DVAL) # Validation scores for Weighted Logistic Regression -> to be calibrated

        case 2:
            model = "Support Vector Machine RBF Kernel"
            kernelFunc = rbfKernel(math.exp(-1))
            C = 1
            KSVM = KernelSupportVectorMachine()
            KSVM.train(DTR, LTR, C, kernelFunc)
            raw_scores = KSVM.get_scores(DVAL) # Validation scores for SVM with RBF Kernel -> to be calibrated

        case 3:
            model = "Gaussian Mixture Model 16 components"
            num_components = 16
            alpha = 0.1
            psi = 0.01
            GMM = GaussianMixtureModel()
            GMM.train(DTR, LTR, numComponents=num_components, alpha=alpha, psi=psi)
            raw_scores = GMM.get_scores(DVAL) # Validation scores for GMM -> to be calibrated

    # Compute calibration transformations for the selected model on the validation set
    # Split scores and labels into K folds
    raw_scores_folds = [raw_scores[i::K] for i in range(K)]
    LVAL_folds = [LVAL[i::K] for i in range(K)]

    # Apply a K-fold cross-validation procedure to compute the optimal logistic regression parameters for calibration (C and K) for each model 
    # TODO: Make this a function and move in "cross_validation.py"
    calibrated_scores = np.zeros_like(raw_scores)

    for k in range(K):
        # Train the calibration model on K-1 folds and validate on the remaining fold
        SCAL, SVAL = np.hstack([raw_scores_folds[i] for i in range(K) if i != k]), raw_scores_folds[k]
        LCAL = np.hstack([LVAL_folds[i] for i in range(K) if i != k])

        # Train calibration model (weighted logistic regression) on the calibration training set with the application prior (pEmp)
        lamb = 1e-3
        WLR = WeightedLogisticRegression()
        WLR.train(vrow(SCAL), LCAL, lamb)

        # Compute calibrated scores on the validation fold
        calibrated_scores[k::K] = WLR.get_log_likelihood_ratios(vrow(SVAL))

    # Compute minDCF, actDCF and calibrated actDCF for the selected model
    # TODO: Generalize, make a function and move to another file
    effPriorLogOdds = np.linspace(-4, 4, 21)
    effPriors = 1.0 / (1.0 + np.exp(-effPriorLogOdds))

    raw_actDCFs = []
    calibrated_actDCFs = []
    minDCFs = []

    print(f"Computing DCFs on raw and calibrated scores of {model}")

    total_iters = len(effPriors)
    for i, effPrior in enumerate(effPriors):
        print(f"Progress: {i / total_iters * 100:.1f}%", end='\r')
        t = -effPriorLogOdds[i]
        # actDCF of raw scores
        PVAL_raw = compute_optimal_bayes_decisions(raw_scores, t)
        raw_actDCFs.append(compute_actual_DCF(effPrior, compute_confusion_matrix(PVAL_raw, LVAL)))

        # minDCF
        minDCFs.append(compute_minimum_DCF(raw_scores, LVAL, effPrior, 1.0, 1.0))

        # actDCF of calibrated scores
        PVAL_cal = compute_optimal_bayes_decisions(calibrated_scores, t)
        calibrated_actDCFs.append(compute_actual_DCF(effPrior, compute_confusion_matrix(PVAL_cal, LVAL)))
    print("Progress: 100.0%")

    # Plot actDCF and minDCF for different values of C for the selected model
    # TODO: Generalize, make a function and move to another file
    plt.figure()
    plt.plot(effPriorLogOdds, raw_actDCFs, label="actDCF (raw)", color='r', linestyle=':')
    plt.plot(effPriorLogOdds, minDCFs, label='minDCF', color='b', linestyle='--')
    plt.plot(effPriorLogOdds, calibrated_actDCFs, label="actDCF (calibrated)", color='g', linestyle='-')
    plt.ylim([0, 0.6])
    plt.xlim([-3, 3])
    plt.title(f"Bayes error plots for {model}")
    plt.ylabel("DCF value")
    plt.xlabel("prior log-odds")
    plt.legend()
    plt.show()