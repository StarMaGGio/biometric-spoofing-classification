# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
from src.models.evaluation import compute_acc_err
from src.models.utils import split_db_2to1
from src.models.visualization import histsPlot, plot_ErrorRate_vs_parameter_function, scatterPlot
from src.models.dimensionality_reduction import PrincipalComponentAnalysis, LinearDiscriminantAnalysis

def analyze_PCA_LDA(D, L):
    
    # --------------------------
    #  Dimensionality Reduction
    # --------------------------

    inner_menu_option = int(input('\nDimensionality Reduction Menu:\n'
                                  '1. Analyze effects of PCA on features\n'
                                  '2. Analyze effects of LDA on features\n'
                                  '3. Apply LDA for classification\n'
                                  '4. Apply PCA + LDA for classification\n'
                                  '0. Back\n'))

    if inner_menu_option == 0: return

    # APPLY LDA FOR CLASSIFICATION
    # Divide the dataset in training and validation sets
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)

    match inner_menu_option:
        case 1:
            # ANALYZE EFFECTS OF PCA ON THE FEATURES
            m = int(input('Number of PCA directions: '))
            PCA = PrincipalComponentAnalysis()
            PCA.train(D, m)
            DP = PCA.apply(D)
            # Plot histograms for the m PCA directions
            histsPlot(DP, L, 'PCA effect on features', m)
        case 2:
            # ANALYZE EFFECTS OF LDA
            m = int(input('Number of LDA directions: '))
            LDA = LinearDiscriminantAnalysis()
            LDA.train(D, L, m)
            DW = LDA.apply(D)
            # Plot histogram
            histsPlot(DW, L, "LDA effect on features", m)
        case 3:
            # Compute and apply LDA matrix to training and validation sets
            m = int(input('Number of LDA directions: '))
            LDA = LinearDiscriminantAnalysis()
            LDA.train(DTR, LTR, m)
            DTRW = LDA.apply(DTR)
            DVALW = LDA.apply(DVAL)
            
            # Compute threshold (in this case the mean of the means of the two classes) for the classification
            threshold = (DTRW[0, LTR==0].mean() + DTRW[0, LTR==1].mean()) / 2.0
            print(f"Threshold: {threshold:.5f}")
            
            # Classify projected DVAL with the threshold computed from projected DTR
            PVAL = np.zeros(shape=LVAL.shape, dtype=np.int32)  
            PVAL[DVALW[0] >= threshold] = 1 # Predict class 1 for elements greater than the threshold
            PVAL[DVALW[0] < threshold] = 0 # Predict class 0 for elements lower than the threshold
            
            # Compute LDA prediction error rate
            acc, err = compute_acc_err(PVAL, LVAL)
            print(f"LDA-only error rate: {err:.5f}")
        case 4:
            # ------ PCA + LDA ------
            error_rates = np.zeros(6)
            for m in range(1, 7):
                # Estimate PCA on initial DTR
                PCA = PrincipalComponentAnalysis()
                PCA.train(DTR, m)
                # Apply PCA on DTR and DVAL
                DTR_pca = PCA.apply(DTR)
                DVAL_pca = PCA.apply(DVAL)

                n = 1
                # Estimate LDA on DTR_pca
                LDA = LinearDiscriminantAnalysis()
                LDA.train(DTR_pca, LTR, n)
                # Apply LDA on DTR_pca and DVAL_pca
                DTR_lda = LDA.apply(DTR_pca)
                DVAL_lda = LDA.apply(DVAL_pca)

                # Estimate threshold from DTR preprocessed with PCA + LDA
                threshold = (DTR_lda[0, LTR==0].mean() + DTR_lda[0, LTR==1].mean()) / 2.0
                print(f"Threshold: {threshold:.5f}")

                # Classify preprocessed DVAL with estimated threshold
                PVAL = np.zeros(shape=LVAL.shape, dtype=np.int32)
                PVAL[DVAL_lda[0] >= threshold] = 1 # Predict class 1 for elements greater than the threshold
                PVAL[DVAL_lda[0] < threshold] = 0 # Predict class 0 for elements lower than the threshold

                # Compute PCA + LDA prediction error rate
                acc, err = compute_acc_err(PVAL, LVAL)
                error_rates[m-1] = err
                print(f"PCA + LDA error rate: {err:.5f}")

            plot_ErrorRate_vs_parameter_function(error_rates, range(1, 7), "Number of PCA directions", "PCA + LDA error rate")
