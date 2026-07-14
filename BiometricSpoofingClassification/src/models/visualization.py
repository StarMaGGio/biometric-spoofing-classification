# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import scipy.stats
from src.models.utils import computeCovariance, vrow, compute_confusion_matrix
from src.models.bayes_decisions_model import compute_actual_DCF, compute_minimum_DCF
from src.models.multivariate_gaussian_log_pdf import logpdf_GAU_ND
from src.models.bayes_decisions_model import compute_optimal_bayes_decisions

def plot_ErrorRate_vs_parameter_function(error_rates, parameters_range, parameter_name, title):
    plt.figure()
    plt.plot(parameters_range, error_rates)
    plt.xlabel(parameter_name)
    plt.ylabel("Error rate")
    plt.title(title)
    plt.show()

def histsPlot(D, L, title, nDimensions = 6):
    hFea = {
        0: "Feature 1",
        1: "Feature 2",
        2: "Feature 3",
        3: "Feature 4",
        4: "Feature 5",
        5: "Feature 6",
        }
    
    D0 = D[:, L==0] # Fake class
    D1 = D[:, L==1] # Genuine class
    
    for idxFea in range(nDimensions):
        plt.figure()
        plt.hist(D0[idxFea, :], bins=10, density=True, alpha=0.4, label='Fake', color="red")
        plt.hist(D1[idxFea, :], bins=10, density=True, alpha=0.4, label='Genuine', color="green")
        plt.title(title)
        plt.xlabel(hFea[idxFea])
        plt.ylabel('Density')
        plt.legend()
        plt.tight_layout()
    plt.show()

def scatterPlot(D, L, idxFea1, idxFea2, nameFea1, nameFea2, title):
    D0 = D[:, L==0] # Fake class
    D1 = D[:, L==1] # Genuine class

    plt.figure()
    plt.scatter(D0[idxFea1, :], D0[idxFea2, :], alpha=0.5, label="Fake", color="red")
    plt.scatter(D1[idxFea1, :], D1[idxFea2, :], alpha=0.5, label="Genuine", color="green")
    plt.xlabel(nameFea1)
    plt.ylabel(nameFea2)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
def scattersPlot(D, L):
    hFea = {
        0: "Feature 1",
        1: "Feature 2",
        2: "Feature 3",
        3: "Feature 4",
        4: "Feature 5",
        5: "Feature 6",
        }
    
    D0 = D[:, L==0] # Fake class
    D1 = D[:, L==1] # Genuine class
    
    for idxFea1 in range(6):
        for idxFea2 in range(6):
            if idxFea1 == idxFea2: continue
            plt.figure()
            plt.scatter(D0[idxFea1, :], D0[idxFea2, :], alpha=0.5, label="Fake", color="red")
            plt.scatter(D1[idxFea1, :], D1[idxFea2, :], alpha=0.5, label="Genuine", color="green")
            plt.xlabel(hFea[idxFea1])
            plt.ylabel(hFea[idxFea2])
            plt.legend()
            plt.tight_layout()
        plt.show()
        
# Plot distibution density on top of the normalized histogram for all the features of the dataset
def plot_distribution_density(D, L, means, covariances, model):
    # For each class, for each feature, compute ML estimate and plot the distibution density on top of the normalized histogram
    XPlot = np.linspace(-4, 4, 1000)
    for c in range(2):
        D_c = D[:, L==c]
        for i in range(D.shape[0]):
            mu_class_fea = means[c][i:i+1]
            if isinstance(covariances, dict):
                C_class_fea = covariances[c][i:i+1, i:i+1]
            else:
                C_class_fea = covariances[i:i+1, i:i+1]
            
            plt.figure()
            plt.hist(D_c[i].ravel(), bins=50, density=True)
            plt.plot(XPlot.ravel(), np.exp(logpdf_GAU_ND(vrow(XPlot), mu_class_fea, C_class_fea)))
            plt.title(f"Gaussian Distribution of Feature {i+1} - Class {c} - Model {model}")
            plt.xlim([-4, 4])
            plt.ylim([0.0, 0.7])
            plt.show()

def plotGaussianDensityEllipseModels(D, L, means, covariances, model):
    D0 = D[:, L==0] # Fake class
    D1 = D[:, L==1] # Genuine class

    feature_pairs = [(f1, f2) for f1, f2 in [(0, 1), (2, 3), (4, 5)] if f2 < D.shape[0]]

    for f1, f2 in feature_pairs:
        
        # --- Extract 2D data for current couple ---
        X0 = D0[[f1, f2], :]
        X1 = D1[[f1, f2], :]

        # --- 2. Extract 2x2 Mean and Covariance Matrices ---
        mu0_2d = means[0][[f1, f2]].reshape(2, 1)
        if isinstance(covariances, dict):
            cov0_2d = covariances[0][np.ix_([f1, f2], [f1, f2])]
            cov1_2d = covariances[1][np.ix_([f1, f2], [f1, f2])]
        else:
            cov0_2d = covariances[np.ix_([f1, f2], [f1, f2])]
            cov1_2d = covariances[np.ix_([f1, f2], [f1, f2])]

        mu1_2d = means[1][[f1, f2]].reshape(2, 1)

        # --- 3. Prepare Grid (Meshgrid) ---
        # Calculate the boundaries of the plot based on the minimum and maximum values of the real data
        x_min = min(X0[0,:].min(), X1[0,:].min()) - 1
        x_max = max(X0[0,:].max(), X1[0,:].max()) + 1
        y_min = min(X0[1,:].min(), X1[1,:].min()) - 1
        y_max = max(X0[1,:].max(), X1[1,:].max()) + 1

        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), 
                            np.linspace(y_min, y_max, 100))
        pos = np.dstack((xx, yy))

        # --- 4. Draw the plot ---
        plt.figure(figsize=(8, 6))

        # Scatter plot of real data points
        plt.scatter(X0[0, :], X0[1, :], label='Class 0 (Fake)', alpha=0.3, s=10)
        plt.scatter(X1[0, :], X1[1, :], label='Class 1 (Genuine)', alpha=0.3, s=10)

        # Calculate Gaussian probabilities on the grid
        grid_coords = np.vstack([xx.ravel(), yy.ravel()])
        pdf0 = np.exp(logpdf_GAU_ND(grid_coords, mu0_2d, cov0_2d)).reshape(xx.shape)
        pdf1 = np.exp(logpdf_GAU_ND(grid_coords, mu1_2d, cov1_2d)).reshape(xx.shape)
        # pdf0 = scipy.stats.multivariate_normal(mu0_2d, cov0_2d).pdf(pos)
        # pdf1 = scipy.stats.multivariate_normal(mu1_2d, cov1_2d).pdf(pos)

        # Draw the ellipse (contour) by drawing the line where the density is at 10% of the maximum peak
        plt.contour(xx, yy, pdf0, levels=[pdf0.max()*0.1], colors='blue', linewidths=2)
        plt.contour(xx, yy, pdf1, levels=[pdf1.max()*0.1], colors='orange', linewidths=2)

        # Labels (add 1 to the indices to make it read "Feature 1" instead of "Feature 0")
        plt.xlabel(f'Feature {f1 + 1}')
        plt.ylabel(f'Feature {f2 + 1}')
        plt.title(f'{model} - 2D Density Ellipses: Feature {f1 + 1} vs Feature {f2 + 1}')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.show()

def plot_Bayes_error(LLRs, LVAL, model_name):
    """
    Function to plot Bayes error for a given model
    
    Args:
        LLRs (np.ndarray): Log-likelihood ratios
        LVAL (np.ndarray): True labels
        model_name (str): Name of the model
    """
    effPriorLogOdds = np.linspace(-4, 4, 21)
    dcf = []
    mindcf = []
    
    # For each prior log-odds, compute actual DCF and minimum DCF
    for p in effPriorLogOdds:
        effPrior = 1/(1+np.exp(-p))
        
        PVAL = compute_optimal_bayes_decisions(LLRs, t=-p)
        
        conf_matr = compute_confusion_matrix(PVAL, LVAL)
        
        DCF = compute_actual_DCF(effPrior, 1, 1, conf_matr)
        dcf.append(DCF)
        
        minDCF = compute_minimum_DCF(LLRs, LVAL, effPrior, 1, 1)
        mindcf.append(minDCF)
        
    # Plot actual DCF and minimum DCF
    plt.figure()
    plt.plot(effPriorLogOdds, dcf, label="DCF", color='r')
    plt.plot(effPriorLogOdds, mindcf, label='min DCF', color='b')
    plt.ylim([0, 1.1])
    plt.xlim([-4, 4])
    plt.title(f"Bayes error plots for {model_name}")
    plt.ylabel("DCF value")
    plt.xlabel("prior log-odds")
    plt.show()

def plot_min_act_actcal_DCF_for_n_systems(raw_scores_list, calibrated_scores_list, LVAL, pi, system_names):
    effPriorLogOdds = np.linspace(-4, 4, 21)
    effPriors = 1.0 / (1.0 + np.exp(-effPriorLogOdds))

    # Print the name of all the systems
    print(f"Computing Bayes Errors on raw scores of {len(raw_scores_list)} systems: {', '.join(system_names)}...")

    rawActDCFs_list = []
    calActDCFs_list = []
    minDCFs_list = []
    
    total_iters = len(effPriors)
    for i, effPrior in enumerate(effPriors):
        print(f"Progress: {i / total_iters * 100:.1f}%", end='\r')

        rawActDCFs = []
        calActDCFs = []
        minDCFs = []
        for raw_scores, calibrated_scores in zip(raw_scores_list, calibrated_scores_list):
            # Compute optimal decisions for raw scores
            PVAL_raw = compute_optimal_bayes_decisions(raw_scores, -effPriorLogOdds[i])
            conf_matr_raw = compute_confusion_matrix(PVAL_raw, LVAL)
            rawActDCFs.append(compute_actual_DCF(effPrior, 1.0, 1.0, conf_matr_raw))
            minDCFs.append(compute_minimum_DCF(raw_scores, LVAL, effPrior, 1.0, 1.0))
            # Compute optimal decisions for calibrated scores
            PVAL_calibrated = compute_optimal_bayes_decisions(calibrated_scores, -effPriorLogOdds[i])
            conf_matr_calibrated = compute_confusion_matrix(PVAL_calibrated, LVAL)
            calActDCFs.append(compute_actual_DCF(effPrior, 1.0, 1.0, conf_matr_calibrated))
        rawActDCFs_list.append(rawActDCFs)
        calActDCFs_list.append(calActDCFs)
        minDCFs_list.append(minDCFs)
    print("Progress: 100.0%")

    colors = ['r', 'b', 'g', 'c', 'm', 'y', 'k']
    plt.figure()
    for i in range(len(system_names)):
        c = colors[i % len(colors)]
        plt.plot(effPriorLogOdds, [rawActDCFs[i] for rawActDCFs in rawActDCFs_list], label=f"{system_names[i]} - actDCF (raw)", color=c, linestyle=':')
        plt.plot(effPriorLogOdds, [calActDCFs[i] for calActDCFs in calActDCFs_list], label=f"{system_names[i]} - actDCF (calibrated)", color=c, linestyle='--')
        plt.plot(effPriorLogOdds, [minDCFs[i] for minDCFs in minDCFs_list], label=f"{system_names[i]} - minDCF", color=c, linestyle='-')
    plt.xlabel('Effective Prior Log Odds')
    plt.ylabel('DCF value')
    plt.title('DCF vs Effective Prior Log Odds for Multiple Systems')
    plt.legend()
    plt.ylim([0, 1.1])
    plt.xlim([-3, 3])
    plt.show()

def plot_min_act_DCF_for_n_systems(scores_list, LVAL, pi, system_names):
    effPriorLogOdds = np.linspace(-4, 4, 21)
    effPriors = 1.0 / (1.0 + np.exp(-effPriorLogOdds))

    # Print the name of all the systems
    print(f"Computing Bayes Errors on scores of {len(scores_list)} systems: {', '.join(system_names)}...")

    actDCFs_list = []
    minDCFs_list = []
    
    total_iters = len(effPriors)
    for i, effPrior in enumerate(effPriors):
        print(f"Progress: {i / total_iters * 100:.1f}%", end='\r')

        actDCFs = []
        minDCFs = []
        for scores in scores_list:
            # Compute optimal decisions for raw scores
            PVAL_raw = compute_optimal_bayes_decisions(scores, -effPriorLogOdds[i])
            conf_matr_raw = compute_confusion_matrix(PVAL_raw, LVAL)
            actDCFs.append(compute_actual_DCF(effPrior, 1.0, 1.0, conf_matr_raw))
            minDCFs.append(compute_minimum_DCF(scores, LVAL, effPrior, 1.0, 1.0))
        actDCFs_list.append(actDCFs)
        minDCFs_list.append(minDCFs)
    print("Progress: 100.0%")

    
    colors = ['r', 'b', 'g', 'c', 'm', 'y', 'k']
    plt.figure()
    for i in range(len(system_names)):
        c = colors[i % len(colors)]
        plt.plot(effPriorLogOdds, [actDCFs[i] for actDCFs in actDCFs_list], label=f"{system_names[i]} - actDCF", color=c, linestyle='--')
        plt.plot(effPriorLogOdds, [minDCFs[i] for minDCFs in minDCFs_list], label=f"{system_names[i]} - minDCF", color=c, linestyle='-')
    plt.xlabel('Effective Prior Log Odds')
    plt.ylabel('DCF value')
    plt.title('DCF vs Effective Prior Log Odds for Multiple Systems')
    plt.legend()
    plt.ylim([0, 1.1])
    plt.xlim([-3, 3])
    plt.show()