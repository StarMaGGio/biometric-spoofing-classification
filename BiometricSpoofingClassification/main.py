# pyrefly: ignore [missing-import]
import numpy as np
import math
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt

from src.models.utils import loadData, split_db_2to1, compute_effective_prior, compute_confusion_matrix, polyKernel, rbfKernel
from src.models.visualization import plot_Bayes_error
from src.models.bayes_decisions_model import compute_actual_DCF, compute_minimum_DCF, compute_optimal_bayes_decisions

from src.models.gaussian_models import MultivariateGaussianClassifier, NaiveBayesGaussianClassifier, TiedGaussianClassifier
from src.models.logistic_regression import LogisticRegression, WeightedLogisticRegression
from src.models.support_vector_machines import SupportVectorMachine, KernelSupportVectorMachine
from src.models.gaussian_mixture_models import GaussianMixtureModel

from src.analysis.analyze_PCA_LDA import analyze_PCA_LDA
from src.analysis.compare_gaussian_models import compare_gaussian_models

# TODO: MOVE ALL THESE FUNCTIONS TO SEPARATE FILES

# -----------------------
#  Evaluation/Bayes Risk
# -----------------------
def compare_effPriors_and_DCFs_for_different_applications(D, L):

    # Divide the dataset in training and validation sets
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)

    inner_menu_option = int(input('\n Choose a model to evaluate:\n'
                                  '1. MVG\n'
                                  '2. Tied Gaussian\n'
                                  '3. Naive Bayes Gaussian\n'
                                  '4. Bayes Error Plot (Different Applications/Effective Priors)\n'
                                  '0. Back\n'))
    model = ""

    if inner_menu_option == 0: return

    if inner_menu_option != 4:
        pi = float(input("Prior probability of genuine sample: "))
        Cfn = float(input("Cost of false negative: "))
        Cfp = float(input("Cost of false positive: "))
        
        # Compute effective prior
        eff_prior = compute_effective_prior(pi, Cfn, Cfp)
        print(f"Application (pi={pi}, Cfn={Cfn}, Cfp={Cfp}) -> Effective Prior: {eff_prior:.2f}")
        t = np.log((1-eff_prior)/eff_prior)

    # Compute optimal Bayes decisions on the validation set for the selected model
    match inner_menu_option:
        case 1:
            model = "MVG"
            MVG = MultivariateGaussianClassifier()
            MVG.train(DTR, LTR)
            PVAL = MVG.predict_binary(DVAL, t)
        case 2:
            model = "Tied Gaussian"
            TG = TiedGaussianClassifier()
            TG.train(DTR, LTR)
            PVAL = TG.predict_binary(DVAL, t)
        case 3:
            model = "Naive Bayes Gaussian"
            NBG = NaiveBayesGaussianClassifier()
            NBG.train(DTR, LTR)
            PVAL = NBG.predict_binary(DVAL, t)
        case 4:
            LLRs = np.zeros(shape=LVAL.shape)
            select_model = int(input("Select model: 1. MVG, 2. Tied Gaussian, 3. Naive Bayes Gaussian"))
            if select_model == 1:
                model = "MVG"
                MVG = MultivariateGaussianClassifier()
                MVG.train(DTR, LTR)
                LLRs = MVG.get_log_likelihood_ratios(DVAL)
            elif select_model == 2:
                model = "Tied Gaussian"
                TG = TiedGaussianClassifier()
                TG.train(DTR, LTR)
                LLRs = TG.get_log_likelihood_ratios(DVAL)
            elif select_model == 3:
                model = "Naive Bayes Gaussian"
                NBG = NaiveBayesGaussianClassifier()
                NBG.train(DTR, LTR)
                LLRs = NBG.get_log_likelihood_ratios(DVAL)
            plot_Bayes_error(LLRs, LVAL, model)
            return

    # Compute minDCF and actDCF
    min_DCF = compute_minimum_DCF(DVAL, LVAL, eff_prior)
    act_DCF = compute_actual_DCF(eff_prior, compute_confusion_matrix(PVAL, LVAL))
    loss = act_DCF - min_DCF
    percent_loss = loss / min_DCF * 100
    print(f"effPrior={eff_prior}: act_DCF={act_DCF:.3f}, min_DCF={min_DCF:.3f}, percent_loss={percent_loss:.3f}")
    
# ---------------------
#  Logistic Regression
# ---------------------
def analyze_logistic_regression_with_different_lambdas(D, L):

    # TODO: Add reduced dataset and quadratic expansion analysis

    # Divide the dataset in training and validation sets
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)

    inner_menu_option = int(input('\n Choose a model to evaluate:\n'
                                  '1. Logistic Regression\n'
                                  '2. Weighted Logistic Regression\n'
                                  '3. Bayes Error Plot (Different Lambdas/Regularization Parameters)\n'
                                  '0. Back\n'))
    model = ""

    if inner_menu_option == 0: return

    if inner_menu_option != 3:
        lamb = float(input("Regularization parameter (from 0.0001 to 100.0): "))

    match inner_menu_option:
        case 1:
            # Train model
            LR = LogisticRegression()
            LR.train(DTR, LTR, lamb)
            # Predict validation labels
            PVAL = LR.predict_binary(DVAL)
        case 2:
            # Train weighted model
            WLR = WeightedLogisticRegression()
            WLR.train(DTR, LTR, lamb)
            # Predict validation labels
            PVAL = WLR.predict_binary(DVAL)
        case 3:
            # Plot DCFs for different values of lambda for the selected model
            LLRs = np.zeros(shape=LVAL.shape)
            select_model = int(input("Select model: 1. Logistic Regression, 2. Weighted Logistic Regression"))
            match select_model:
                case 1:
                    model = "Logistic Regression"
                    
                    actDCFs = []
                    minDCFs = []
                    lambs = np.logspace(-4, 2, 13)
                    effPrior = 0.1
                    for lamb in lambs:
                        LR = LogisticRegression()
                        LR.train(DTR, LTR, lamb)
                        LLRs = LR.get_log_likelihood_ratios(DVAL)

                        minDCF = compute_minimum_DCF(LLRs, LVAL, effPrior, 1.0, 1.0)
                        minDCFs.append(minDCF)
                        print('minDCF: %.4f' % minDCF)

                        conf_matr = compute_confusion_matrix(PVAL, LVAL)
                        actDCF = compute_actual_DCF(effPrior, 1.0, 1.0, conf_matr)
                        actDCFs.append(actDCF)
                        print('actDCF: %.4f' % actDCF)
                    
                        print()
                case 2:
                    model = "Weighted Logistic Regression"

                    actDCFs = []
                    minDCFs = []
                    lambs = np.logspace(-4, 2, 13)
                    effPrior = 0.1
                    for lamb in lambs:
                        WLR = WeightedLogisticRegression()
                        WLR.train(DTR, LTR, lamb)
                        LLRs = WLR.get_log_likelihood_ratios(DVAL)

                        minDCF = compute_minimum_DCF(LLRs, LVAL, effPrior, 1.0, 1.0)
                        minDCFs.append(minDCF)
                        print('minDCF: %.4f' % minDCF)

                        conf_matr = compute_confusion_matrix(PVAL, LVAL)
                        actDCF = compute_actual_DCF(effPrior, 1.0, 1.0, conf_matr)
                        actDCFs.append(actDCF)
                        print('actDCF: %.4f' % actDCF)
                    
                        print()
                
            plt.figure()
            plt.plot(lambs, minDCFs, label="minDCF", color='r')
            plt.plot(lambs, actDCFs, label="actDCF", color='b')
            plt.xscale('log', base=10)
            plt.ylabel('DCF value')
            plt.xlabel('Lambda / Regularization')
            plt.title(f"{model} - Different Lambdas/Regularization Parameters")
            plt.show()
            return

    # Compute error rate
    err = (PVAL != LVAL).sum() / float(LVAL.size)
    print('Error rate: %.2f' % (err*100))

# -------------------------
#  Support Vector Machines
# -------------------------
def analyze_SVM_with_different_kernels(D, L):

    # Divide the dataset in training and validation sets
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)

    inner_menu_option = int(input('\n Choose a model to evaluate:\n'
                                  '1. Linear Support Vector Machine\n'
                                  '2. Linear Support Vector Machine (Centered Data)\n'
                                  '3. Support Vector Machine Polynomial Kernel\n'
                                  '4. Support Vector Machine RBF Kernel\n'
                                  '0. Back\n'))
    model = ""

    if inner_menu_option == 0: return

    dataset_portion = int(input('Database portion to use for training and validation (1-100): ')) 
    
    # Check if dataset_portion is within the valid range
    if not (1 <= dataset_portion <= 100):
        print("Error: dataset_portion must be between 1 and 100.")
        return

    DTR = DTR[:, ::dataset_portion]
    LTR = LTR[::dataset_portion]
    DVAL = DVAL[:, ::dataset_portion]
    LVAL = LVAL[::dataset_portion]
    minDCFs = []
    actDCFs = []

    match inner_menu_option:
        case 1:
            model = "SVM Linear"
            Cs = np.logspace(-5, 0, 11)   
            minDCFs.clear()
            actDCFs.clear()
            for C in Cs:
                SVM = SupportVectorMachine()
                SVM.train(DTR, LTR, C, K=1.0)
                minDCFs.append(compute_minimum_DCF(SVM.get_scores(DVAL), LVAL, 0.1, 1.0, 1.0))
                actDCFs.append(compute_actual_DCF(0.1, 1.0, 1.0, compute_confusion_matrix(SVM.predict(DVAL), LVAL)))
        case 2:
            model = "SVM Linear Centered Data"
            Cs = np.logspace(-5, 0, 11)   
            minDCFs.clear()
            actDCFs.clear()
            # Center Dataset
            mu = DTR.mean(1).reshape((DTR.shape[0], 1))
            DTR_centered = DTR - mu
            DVAL_centered = DVAL - mu
            for C in Cs:
                SVM = SupportVectorMachine()
                SVM.train(DTR_centered, LTR, C, K=1.0)
                minDCFs.append(compute_minimum_DCF(SVM.get_scores(DVAL_centered), LVAL, 0.1, 1.0, 1.0))
                actDCFs.append(compute_actual_DCF(0.1, 1.0, 1.0, compute_confusion_matrix(SVM.predict(DVAL_centered), LVAL)))
        case 3:
            degree = int(input('Degree of the polynomial kernel (ex. 2): '))
            model = f"SVM Polynomial Kernel (degree = {degree})"
            kernelFunc = polyKernel(degree, 1)
            eps = 0.0
            Cs = np.logspace(-5, 0, 11)
            minDCFs.clear()
            actDCFs.clear()
            for C in Cs:
                KSVM = KernelSupportVectorMachine()
                KSVM.train(DTR, LTR, C, kernelFunc=kernelFunc, eps=eps)
                minDCFs.append(compute_minimum_DCF(KSVM.get_scores(DVAL), LVAL, 0.1, 1.0, 1.0))
                actDCFs.append(compute_actual_DCF(0.1, 1.0, 1.0, compute_confusion_matrix(KSVM.predict(DVAL), LVAL)))
        case 4:
            gamma = float(input('Gamma parameter for the RBF kernel (0.0001 - 10): '))
            model = f"SVM RBF Kernel (gamma = {gamma})"
            minDCFs.clear()
            actDCFs.clear()
            eps = 1.0
            Cs = np.logspace(-3, 2, 11)
            kernelFunc = rbfKernel(gamma)
            for C in Cs:
                KSVM = KernelSupportVectorMachine()
                KSVM.train(DTR, LTR, C, kernelFunc=kernelFunc, eps=eps)
                minDCFs.append(compute_minimum_DCF(KSVM.get_scores(DVAL), LVAL, 0.1, 1.0, 1.0))
                actDCFs.append(compute_actual_DCF(0.1, 1.0, 1.0, compute_confusion_matrix(KSVM.predict(DVAL), LVAL)))

    # Plot actDCF and minDCF for different values of C for the selected model
    plt.figure()
    plt.plot(Cs, minDCFs, label="minDCF", color='r')
    plt.plot(Cs, actDCFs, label="actDCF", color='b')
    plt.xscale('log', base=10)
    plt.ylabel('DCF value')
    plt.xlabel('C value')
    plt.title(f"{model} - {dataset_portion}% of data")
    plt.legend()
    plt.show()
    print()
    
# -------------------------
#  Gaussian Mixture Models
# -------------------------
def analyze_GMM_with_different_components(D, L):
    # Divide the dataset in training and validation sets
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)

    inner_menu_option = int(input('\n Choose a model to evaluate:\n'
                                  '1. Gaussian Mixture Model\n'
                                  '0. Back\n'))
    model = ""

    if inner_menu_option == 0: return

    match inner_menu_option:
        case 1:
            num_components = int(input("Enter the number of components for the GMM (1, 2, 4, 8, 16): "))
            alpha = float(input("Enter the alpha parameter for the GMM (ex 0.1): "))
            psi = float(input("Enter the psi parameter for the GMM (ex. 0.01): "))

            eff_prior = 0.1
            t = np.log((1-eff_prior)/eff_prior)

            GMM = GaussianMixtureModel()
            GMM.train(DTR, LTR, numComponents=num_components, alpha=alpha, psi=psi)
            PVAL = GMM.predict(DVAL, t)

            print(f"Components: {num_components}: actual DCF: {compute_actual_DCF(eff_prior, 1.0, 1.0, compute_confusion_matrix(PVAL, LVAL)):.4f}")

# --------------------
#  Scores Calibration
# --------------------
def scores_calibration(D, L):
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
            model = "Weighted Logistic Regression"
            lamb = 10 ** -1.5
            WLR = WeightedLogisticRegression()
            WLR.train(DTR, LTR, lamb)
            raw_scores = WLR.get_log_likelihood_ratios(DVAL) # Validation scores for Weighted Logistic Regression -> to be calibrated
        
        case 2:
            model = "Support Vector Machine RBF Kernel"
            kernelFunc = rbfKernel(math.exp(-2))
            C = 10 ** 1.5
            KSVM = KernelSupportVectorMachine()
            KSVM.train(DTR, LTR, C, kernelFunc)
            raw_scores = KSVM.get_scores(DVAL) # Validation scores for SVM with RBF Kernel -> to be calibrated

        case 3:
            model = "Gaussian Mixture Model 8 components"
            num_components = 8
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
        LCAL, LVAL = np.hstack([LVAL_folds[i] for i in range(K) if i != k]), LVAL_folds[k]

        # Train calibration model (weighted logistic regression) on the calibration training set with the application prior (pEmp)
        lamb = 1e-3
        WLR = WeightedLogisticRegression()
        WLR.train(SCAL, LCAL, lamb)

        # Compute calibrated scores on the validation fold
        calibrated_scores[k::K] = WLR.get_log_likelihood_ratios(SVAL)

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
        # actDCF of raw scores
        PVAL_raw = compute_optimal_bayes_decisions(raw_scores, effPrior)
        raw_actDCFs.append(compute_actual_DCF(effPrior, 1.0, 1.0, compute_confusion_matrix(PVAL_raw, LVAL)))

        # minDCF
        minDCFs.append(compute_minimum_DCF(raw_scores, LVAL, effPrior, 1.0, 1.0))

        # actDCF of calibrated scores
        PVAL_cal = compute_optimal_bayes_decisions(calibrated_scores, effPrior)
        calibrated_actDCFs.append(compute_actual_DCF(effPrior, 1.0, 1.0, compute_confusion_matrix(PVAL_cal, LVAL)))
    print("Progress: 100.0%")

    # Plot actDCF and minDCF for different values of C for the selected model
    # TODO: Generalize, make a function and move to another file
    plt.figure()
    plt.plot(effPriorLogOdds, raw_actDCFs, label="actDCF (raw)", color='r', linestyle=':')
    plt.plot(effPriorLogOdds, minDCFs, label='minDCF', color='b', linestyle='--')
    plt.plot(effPriorLogOdds, calibrated_actDCFs, label="actDCF (calibrated)", color='g', linestyle='-')
    plt.ylim([0, 1.1])
    plt.xlim([-3, 3])
    plt.title(f"Bayes error plots for {model}")
    plt.ylabel("DCF value")
    plt.xlabel("prior log-odds")
    plt.legend()
    plt.show()

# ---------------------
#  Scores Level Fusion
# ---------------------
def score_level_fusion(D, L):

    # Divide the dataset in training and validation sets
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)

    print("\nTraining models for score fusion")

    # Train three different models (Logistic Regression, Support Vector Machine, Gaussian Mixture Model)
    lamb = 10 ** -1.5
    WLR = WeightedLogisticRegression()
    WLR.train(DTR, LTR, lamb)
    raw_scores_lr = WLR.get_log_likelihood_ratios(DVAL) # Raw validation scores for Weighted Logistic Regression

    kernelFunc = rbfKernel(math.exp(-2))
    C = 10 ** 1.5
    KSVM = KernelSupportVectorMachine()
    KSVM.train(DTR, LTR, C, kernelFunc)
    raw_scores_svm = KSVM.get_scores(DVAL) # Raw validation scores for SVM with RBF Kernel

    num_components = 8
    alpha = 0.1
    psi = 0.01
    GMM = GaussianMixtureModel()
    GMM.train(DTR, LTR, numComponents=num_components, alpha=alpha, psi=psi)
    raw_scores_gmm = GMM.get_scores(DVAL) # Raw validation scores for GMM with 8 components

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

        # Train calibration model (weighted logistic regression) on the calibration training set with the application prior (pEmp)
        lamb = 1e-3
        WLR = WeightedLogisticRegression()
        WLR.train(SCAL, LCAL, lamb)

        # Compute calibrated scores on the validation fold
        calibrated_scores_fusion[:, k::K] = WLR.get_log_likelihood_ratios(SVAL_k)
    
    print("\n Computing DCFs on calibrated scores for the fused system")
    # Compute and print DCF for the fused system
    plot_Bayes_error(calibrated_scores_fusion, LVAL, "Fused System (Weighted Logistic Regression)")
    
# --------------------
#  Evaluation Dataset
# --------------------
def final_evaluation(D, L):
    
    # --- Full Pipeline ---
    # --- PHASE 0: Divide the dataset in training and validation sets
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)

    # --- PHASE 1: Obtain Raw Scores for the three models (K-Fold on DTR) ---
    lamb = 10 ** -1.5
    WLR = WeightedLogisticRegression()
    WLR.train(DTR, LTR, lamb)
    val_scores_LR = WLR.get_log_likelihood_ratios(DVAL) # Raw validation scores for Weighted Logistic Regression

    kernelFunc = rbfKernel(math.exp(-2))
    C = 10 ** 1.5
    KSVM = KernelSupportVectorMachine()
    KSVM.train(DTR, LTR, C, kernelFunc)
    val_scores_SVM = KSVM.get_scores(DVAL) # Raw validation scores for SVM with RBF Kernel

    num_components = 8
    alpha = 0.1
    psi = 0.01
    GMM = GaussianMixtureModel()
    GMM.train(DTR, LTR, numComponents=num_components, alpha=alpha, psi=psi)
    val_scores_GMM = GMM.get_scores(DVAL) # Raw validation scores for GMM with 8 components

    # --- PHASE 2: K-Fold to Evaluate Calibration Effectiveness
    K = 5
    val_scores_LR_folds = [val_scores_LR[:, i::K] for i in range(K)]
    val_scores_SVM_folds = [val_scores_SVM[:, i::K] for i in range(K)]
    val_scores_GMM_folds = [val_scores_GMM[:, i::K] for i in range(K)]
    LVAL_folds = [LVAL[i::K] for i in range(K)]
    cal_scores_LR = np.zeros_like(val_scores_LR[0])
    cal_scores_SVM = np.zeros_like(val_scores_SVM[0])
    cal_scores_GMM = np.zeros_like(val_scores_GMM[0])
    
    for k in range(K):
        # Calibration Training (Train on K-1 folds)
        SCAL_lr, SVAL_lr = np.hstack([val_scores_LR_folds[i] for i in range(K) if i != k]), val_scores_LR_folds[k]
        LCAL_lr, LVAL_lr = np.hstack([LVAL_folds[i] for i in range(K) if i != k]), LVAL_folds[k]
        SCAL_svm, SVAL_svm = np.hstack([val_scores_SVM_folds[i] for i in range(K) if i != k]), val_scores_SVM_folds[k]
        LCAL_svm, LVAL_svm = np.hstack([LVAL_folds[i] for i in range(K) if i != k]), LVAL_folds[k]
        SCAL_gmm, SVAL_gmm = np.hstack([val_scores_GMM_folds[i] for i in range(K) if i != k]), val_scores_GMM_folds[k]
        LCAL_gmm, LVAL_gmm = np.hstack([LVAL_folds[i] for i in range(K) if i != k]), LVAL_folds[k]

        # Calibration using Weighted Logistic Regression
        pEmp = (LCAL_lr == 1).sum() / LCAL_lr.size
        l = 1e-3
        WLR_Calibrator = WeightedLogisticRegression()
        WLR_Calibrator.train(SCAL_lr, LCAL_lr, l)
        calibrated_scores_lr = WLR_Calibrator.get_log_likelihood_ratios(SVAL_lr)
        
        WLR_Calibrator = WeightedLogisticRegression()
        WLR_Calibrator.train(SCAL_svm, LCAL_svm, l)
        calibrated_scores_svm = WLR_Calibrator.get_log_likelihood_ratios(SVAL_svm)
        
        WLR_Calibrator = WeightedLogisticRegression()
        WLR_Calibrator.train(SCAL_gmm, LCAL_gmm, l)
        calibrated_scores_gmm = WLR_Calibrator.get_log_likelihood_ratios(SVAL_gmm)

        cal_scores_LR[:, k::K] = calibrated_scores_lr
        cal_scores_SVM[:, k::K] = calibrated_scores_svm
        cal_scores_GMM[:, k::K] = calibrated_scores_gmm

    # --- PHASE 3: Prepare the final system (Calibrators and Score-level Fusion)
    Final_WLR_Calibrator_LR = WeightedLogisticRegression()
    Final_WLR_Calibrator_SVM = WeightedLogisticRegression()
    Final_WLR_Calibrator_GMM = WeightedLogisticRegression()
    
    Final_WLR_Calibrator_LR.train(val_scores_LR, LVAL, 1e-3)
    Final_WLR_Calibrator_SVM.train(val_scores_SVM, LVAL, 1e-3)
    Final_WLR_Calibrator_GMM.train(val_scores_GMM, LVAL, 1e-3)

    # Compute calibrated scores for the fused system
    cal_scores_LR_final = Final_WLR_Calibrator_LR.get_log_likelihood_ratios(val_scores_LR)
    cal_scores_SVM_final = Final_WLR_Calibrator_SVM.get_log_likelihood_ratios(val_scores_SVM)
    cal_scores_GMM_final = Final_WLR_Calibrator_GMM.get_log_likelihood_ratios(val_scores_GMM)

    # train fusion model on cal scores

    # --- PHASE 4: Final evaluation on DEVAL
    # 1. extract DEVAL raw scores using base models
    # 2. apply final calibrators
    # 3. pass calibrated scores to fusion model -> fused scores
    # 4. plot_Bayes_error
    

if __name__ == "__main__":

    np.set_printoptions(precision=3, suppress=True)
    D, L = loadData("data/trainData.txt")
    
    while True:
        menu_option = int(input("\nMenu\n"
                                "1. Dimensionality Reduction\n"
                                "2. Generative Gaussian Models\n"
                                "3. Evaluate Gaussian Models with DCFs\n"
                                "4. Logistic Regression\n"
                                "5. Support Vector Machines\n"
                                "6. Gaussian Mixture Models\n"
                                "7. Scores Calibration\n"
                                "8. Score Level Fusion\n"
                                "9. Final Models on Evaluation Dataset \n"
                                "0. Exit\n"))

        match menu_option:
            case 1:
                analyze_PCA_LDA(D, L)
            case 2:
                compare_gaussian_models(D, L)
            case 3:
                compare_effPriors_and_DCFs_for_different_applications(D, L)
            case 4:
                analyze_logistic_regression_with_different_lambdas(D, L)
            case 5:
                analyze_SVM_with_different_kernels(D, L)
            case 6:
                analyze_GMM_with_different_components(D, L)
            case 7:
                scores_calibration(D, L)
            case 8:
                score_level_fusion(D, L)
            case 9:
                final_evaluation(D, L)
            case 0:
                break