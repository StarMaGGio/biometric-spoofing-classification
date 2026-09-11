# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]

from src.models.utils import loadData

from src.analysis.analyze_PCA_LDA import analyze_PCA_LDA
from src.analysis.compare_gaussian_models import compare_gaussian_models
from src.analysis.compare_effPriors_and_DCFs_for_different_applications import compare_effPriors_and_DCFs_for_different_applications
from src.analysis.analyze_logistic_regression_with_different_lambdas import analyze_logistic_regression_with_different_lambdas
from src.analysis.analyze_SVM_with_different_kernels import analyze_SVM_with_different_kernels
from src.analysis.analyze_GMM_with_different_components import analyze_GMM_with_different_components
from src.analysis.scores_calibration import scores_calibration
from src.analysis.score_level_fusion import score_level_fusion
from src.analysis.final_evaluation import final_evaluation

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
                DEVAL, LEVAL = loadData("data/evalData.txt")
                final_evaluation(D, L, DEVAL, LEVAL)
            case 0:
                break