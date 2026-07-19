from src.models.utils import split_db_2to1, computeCovariance, computeCorrelationMatrix
from src.models.evaluation import compute_acc_err
from src.models.dimensionality_reduction import PrincipalComponentAnalysis
from src.models.gaussian_models import MultivariateGaussianClassifier, NaiveBayesGaussianClassifier, TiedGaussianClassifier
from src.models.visualization import plot_distribution_density, plotGaussianDensityEllipseModels, plotMatrix, plot_ErrorRate_vs_parameter_function

def compare_gaussian_models(D, L):

    # ----------------------------
    #  Generative Gaussian Models
    # ----------------------------

    inner_menu_option = int(input('\n Generative Gaussian Models Menu:\n'
                                  '1. Multivariate Gaussian Classifier\n'
                                  '2. Naive Bayes Gaussian Classifier\n'
                                  '3. Tied Gaussian Classifier\n'
                                  '0. Back\n'))

    if inner_menu_option == 0: return

    first_feature, last_feature = int(input("\nFeatures range to consider (from 1 to 6): "))-1, int(input("to "))
    D_sel = D[first_feature:last_feature, :]
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D_sel, L)

    pca_selection = int(input("\nPreprocessing with PCA? (1 for yes, 0 for no): "))
    match pca_selection:
        case 1:
            m = int(input("\nNumber of PCA directions: "))
            PCA = PrincipalComponentAnalysis()
            PCA.train(DTR, m)
            DTR = PCA.apply(DTR)
            DVAL = PCA.apply(DVAL)
        case 0:
            pass

    match inner_menu_option:
        case 1:
            # --- MVG ---
            MVG = MultivariateGaussianClassifier()
            MVG.train(DTR, LTR)
            PVAL = MVG.predict_binary(DVAL)
            acc, err = compute_acc_err(PVAL, LVAL)
            print(f"MVG error rate - features {first_feature+1} to {last_feature}: {err:.5f}")
            # plot_distribution_density(D_sel, L, MVG.means, MVG.covariances, "MVG")
            # plotGaussianDensityEllipseModels(D_sel, L, MVG.means, MVG.covariances, "MVG")
        case 2:
            # --- Naive Bayes Gaussian ---
            NBG = NaiveBayesGaussianClassifier()
            NBG.train(DTR, LTR)
            PVAL = NBG.predict_binary(DVAL)
            acc, err = compute_acc_err(PVAL, LVAL)
            print(f"Naive Bayes Gaussian error rate - features {first_feature+1} to {last_feature}: {err:.5f}")
            # plot_distribution_density(D_sel, L, NBG.means, NBG.covariances, "NBG")
            # plotGaussianDensityEllipseModels(D_sel, L, NBG.means, NBG.covariances, "NBG")
        case 3:
            # --- Tied Gaussian ---
            TG = TiedGaussianClassifier()
            TG.train(DTR, LTR)
            PVAL = TG.predict_binary(DVAL)
            acc, err = compute_acc_err(PVAL, LVAL)
            print(f"Tied Gaussian error rate - features {first_feature+1} to {last_feature}: {err:.5f}")
            # plot_distribution_density(D_sel, L, TG.means, TG.Sw, "TG")
            # plotGaussianDensityEllipseModels(D_sel, L, TG.means, TG.Sw, "TG")
        case 4:
            error_rates = []
            for m in range(1, 7):
                PCA = PrincipalComponentAnalysis()
                PCA.train(DTR, m)
                DTR_pca = PCA.apply(DTR)
                DVAL_pca = PCA.apply(DVAL)

                MVG = MultivariateGaussianClassifier()
                MVG.train(DTR_pca, LTR)
                PVAL = MVG.predict_binary(DVAL_pca)
                acc, err = compute_acc_err(PVAL, LVAL)
                error_rates.append(err)
            plot_ErrorRate_vs_parameter_function(error_rates, range(1, 7), "Number of PCA directions", "PCA + MVG error rate")

            error_rates = []
            for m in range(1, 7):
                PCA = PrincipalComponentAnalysis()
                PCA.train(DTR, m)
                DTR_pca = PCA.apply(DTR)
                DVAL_pca = PCA.apply(DVAL)

                NBG = NaiveBayesGaussianClassifier()
                NBG.train(DTR_pca, LTR)
                PVAL = NBG.predict_binary(DVAL_pca)
                acc, err = compute_acc_err(PVAL, LVAL)
                error_rates.append(err)
            plot_ErrorRate_vs_parameter_function(error_rates, range(1, 7), "Number of PCA directions", "PCA + Naive Bayes Gaussian error rate")

            error_rates = []
            for m in range(1, 7):
                PCA = PrincipalComponentAnalysis()
                PCA.train(DTR, m)
                DTR_pca = PCA.apply(DTR)
                DVAL_pca = PCA.apply(DVAL)

                TG = TiedGaussianClassifier()
                TG.train(DTR_pca, LTR)
                PVAL = TG.predict_binary(DVAL_pca)
                acc, err = compute_acc_err(PVAL, LVAL)
                error_rates.append(err)
            plot_ErrorRate_vs_parameter_function(error_rates, range(1, 7), "Number of PCA directions", "PCA + Tied Gaussian error rate")

            
