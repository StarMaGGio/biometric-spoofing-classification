# pyrefly: ignore [missing-import]
import numpy as np
import math
# pyrefly: ignore [missing-import]

from src.models.utils import loadData, split_db_2to1, rbfKernel, vrow
from src.models.visualization import plot_Bayes_error

from src.models.logistic_regression import WeightedLogisticRegression
from src.models.support_vector_machines import KernelSupportVectorMachine
from src.models.gaussian_mixture_models import GaussianMixtureModel

from src.analysis.analyze_PCA_LDA import analyze_PCA_LDA
from src.analysis.compare_gaussian_models import compare_gaussian_models
from src.analysis.compare_effPriors_and_DCFs_for_different_applications import compare_effPriors_and_DCFs_for_different_applications
from src.analysis.analyze_logistic_regression_with_different_lambdas import analyze_logistic_regression_with_different_lambdas
from src.analysis.analyze_SVM_with_different_kernels import analyze_SVM_with_different_kernels
from src.analysis.analyze_GMM_with_different_components import analyze_GMM_with_different_components
from src.analysis.scores_calibration import scores_calibration

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
        calibrated_scores_fusion[k::K] = WLR.get_log_likelihood_ratios(SVAL_k)
    
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
    val_scores_LR_folds = [val_scores_LR[i::K] for i in range(K)]
    val_scores_SVM_folds = [val_scores_SVM[i::K] for i in range(K)]
    val_scores_GMM_folds = [val_scores_GMM[i::K] for i in range(K)]
    LVAL_folds = [LVAL[i::K] for i in range(K)]
    cal_scores_LR = np.zeros_like(val_scores_LR)
    cal_scores_SVM = np.zeros_like(val_scores_SVM)
    cal_scores_GMM = np.zeros_like(val_scores_GMM)
    
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
        WLR_Calibrator.train(vrow(SCAL_lr), LCAL_lr, l)
        calibrated_scores_lr = WLR_Calibrator.get_log_likelihood_ratios(vrow(SVAL_lr))
        
        WLR_Calibrator = WeightedLogisticRegression()
        WLR_Calibrator.train(vrow(SCAL_svm), LCAL_svm, l)
        calibrated_scores_svm = WLR_Calibrator.get_log_likelihood_ratios(vrow(SVAL_svm))
        
        WLR_Calibrator = WeightedLogisticRegression()
        WLR_Calibrator.train(vrow(SCAL_gmm), LCAL_gmm, l)
        calibrated_scores_gmm = WLR_Calibrator.get_log_likelihood_ratios(vrow(SVAL_gmm))

        cal_scores_LR[k::K] = calibrated_scores_lr
        cal_scores_SVM[k::K] = calibrated_scores_svm
        cal_scores_GMM[k::K] = calibrated_scores_gmm

    # --- PHASE 3: Prepare the final system (Calibrators and Score-level Fusion)
    Final_WLR_Calibrator_LR = WeightedLogisticRegression()
    Final_WLR_Calibrator_SVM = WeightedLogisticRegression()
    Final_WLR_Calibrator_GMM = WeightedLogisticRegression()
    
    Final_WLR_Calibrator_LR.train(vrow(val_scores_LR), LVAL, 1e-3)
    Final_WLR_Calibrator_SVM.train(vrow(val_scores_SVM), LVAL, 1e-3)
    Final_WLR_Calibrator_GMM.train(vrow(val_scores_GMM), LVAL, 1e-3)

    # Compute calibrated scores for the fused system
    cal_scores_LR_final = Final_WLR_Calibrator_LR.get_log_likelihood_ratios(vrow(val_scores_LR))
    cal_scores_SVM_final = Final_WLR_Calibrator_SVM.get_log_likelihood_ratios(vrow(val_scores_SVM))
    cal_scores_GMM_final = Final_WLR_Calibrator_GMM.get_log_likelihood_ratios(vrow(val_scores_GMM))

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