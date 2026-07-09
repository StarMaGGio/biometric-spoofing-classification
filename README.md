# Biometric Spoofing Classification 🤖

This repository contains the project developed for the **Machine Learning and Pattern Recognition** course at **Politecnico di Torino**. 
The objective is to implement machine learning algorithms from scratch (using only `numpy` and `scipy` for mathematical computations) and apply them to a binary classification task: detecting biometric spoofing (distinguishing "Genuine" from "Fake" biometric samples).

---

## 📂 Project Structure

The project is structured under the `BiometricSpoofingClassification` directory:

```
Project/
├── README.md
└── BiometricSpoofingClassification/
    ├── main.py                     # Main CLI entry point with execution menu
    ├── .gitignore                  # Git ignore rules for the subfolder
    ├── data/                       # Dataset folder
    │   ├── trainData.txt           # Training and validation dataset
    │   └── evalData.txt            # Final evaluation dataset
    └── src/                        # Source folder containing module implementations
        ├── utils.py                # Math utilities, data loaders, split/confusion functions, and kernels
        ├── dimensionality_reduction.py # PCA & LDA implementation
        ├── gaussian_models.py      # Multivariate Gaussian, Naive Bayes, and Tied Gaussian Classifiers
        ├── logistic_regression.py  # Standard and Prior-Weighted Logistic Regression
        ├── support_vector_machines.py # Linear and Kernel SVM (using L-BFGS-B optimization)
        ├── gaussian_mixture_models.py # GMM with LBG initialization and EM parameter estimation
        ├── multivariate_gaussian_log_pdf.py # Vectorized log-density computation for Gaussian models
        ├── bayes_decisions_model.py # Bayes decisions and Detection Cost Function (DCF) utilities
        ├── evaluation.py           # Metrics computation (accuracy, error rate)
        └── visualization.py        # Plotting functions (histograms, scatter, Bayes error, DCF curves)
```

### Module Breakdown:
* **[main.py]**: Orchestrates the pipeline, letting the user interactively select and run preprocessing, train classifiers, run scores calibration/fusion, and perform final model evaluations.
* **[utils.py]**: Handles basic operations like reshaping (`vcol`/`vrow`), loading files, dataset partitioning, confusion matrix calculations, and polynomial/RBF kernel functions.
* **[dimensionality_reduction.py]**: Contains `PrincipalComponentAnalysis` (PCA) and `LinearDiscriminantAnalysis` (LDA).
* **[gaussian_models.py]**: Implements generative models, specifically MVG, Tied MVG, and Naive Bayes Classifiers.
* **[logistic_regression.py]**: Regularized Logistic Regression and Prior-Weighted Logistic Regression (for prior compensation).
* **[support_vector_machines.py]**: Soft-margin linear SVM and Kernel SVM (supporting customized kernel functions).
* **[gaussian_mixture_models.py]**: Full GMM pipeline featuring the Linde-Buzo-Gray (LBG) splitting algorithm and Expectation-Maximization (EM) optimization.
* **[multivariate_gaussian_log_pdf.py]**: Vectorized computations of the log probability density functions.
* **[bayes_decisions_model.py]**: Computes Bayes risk, minimum DCF (minDCF), actual DCF (actDCF) and maps likelihood scores to binary classifications.
* **[evaluation.py]**: Computes metrics (accuracy, error rate).
* **[visualization.py]**: Displays feature histograms, scatter plots, model-specific Bayes error plots, and comparison graphs.

---

## 📝 TODOs & Recommended Modifications

Below is a consolidated list of code TODOs left in the project, followed by critical bugs and architectural improvements identified in the codebase.

### 🔍 Code TODOs (from `main.py`)
1. **Pipeline Modularity** (Line 19):
   - *Task*: Move all the pipeline analysis functions defined in [main.py](file:///c:/Users/matti/Documents/PoliTO/Machine%20Learning%20and%20Pattern%20Recognition/Project/BiometricSpoofingClassification/main.py) (e.g., `PCA_LDA_effects_and_classification_analysis`, `compare_gaussian_models`, etc.) to separate files to keep the main script clean.
2. **Logistic Regression Enhancements** (Line 239):
   - *Task*: Add reduced dataset analysis and quadratic feature expansion options to [analyze_logistic_regression_with_different_lambdas](file:///c:/Users/matti/Documents/PoliTO/Machine%20Learning%20and%20Pattern%20Recognition/Project/BiometricSpoofingClassification/main.py#L237) function.
3. **Cross-Validation Function for Calibration** (Line 508):
   - *Task*: Refactor the K-Fold score calibration loop into a reusable function in a new `cross_validation.py` file.
4. **Generalize DCF Computations** (Line 525):
   - *Task*: Generalize the loop computing actDCF/minDCF over different effective priors, package it into a utility function, and move it to a source module.
5. **Generalize Bayes Error Plotting** (Line 551):
   - *Task*: Refactor the inline plotting logic for Bayes error curve into a function in `visualization.py`.
6. **Cross-Validation Function for Fusion** (Line 599):
   - *Task*: Extract the K-Fold fusion calibration loop into the `cross_validation.py` module.

### 🛠️ Recommended Code & Design Fixes
During code analysis, several bugs, broken imports, and logical defects were identified. Fixing these is highly recommended to prevent runtime crashes:

1. **Undefined Function `compute_predictions_with_llr`**:
   - **Issue**: Imported in multiple modules (e.g., [gaussian_models.py](file:///c:/Users/matti/Documents/PoliTO/Machine%20Learning%20and%20Pattern%20Recognition/Project/BiometricSpoofingClassification/src/gaussian_models.py#L8), [logistic_regression.py](file:///c:/Users/matti/Documents/PoliTO/Machine%20Learning%20and%20Pattern%20Recognition/Project/BiometricSpoofingClassification/src/logistic_regression.py#L6), and [gaussian_mixture_models.py](file:///c:/Users/matti/Documents/PoliTO/Machine%20Learning%20and%20Pattern%20Recognition/Project/BiometricSpoofingClassification/src/gaussian_mixture_models.py#L6)), but **never defined** inside `src/bayes_decisions_model.py`.
   - **Fix**: Re-route imports/calls to [compute_optimal_bayes_decisions](file:///c:/Users/matti/Documents/PoliTO/Machine%20Learning%20and%20Pattern%20Recognition/Project/BiometricSpoofingClassification/src/bayes_decisions_model.py#L74) or define the missing helper.
2. **Incorrect Module Path in Imports**:
   - **Issue**: [gaussian_models.py](file:///c:/Users/matti/Documents/PoliTO/Machine%20Learning%20and%20Pattern%20Recognition/Project/BiometricSpoofingClassification/src/gaussian_models.py#L8) includes: `from Project.src.bayes_decisions_model import ...`. Since the project root folder is `BiometricSpoofingClassification`, imports starting with `Project` will raise an `ImportError`.
   - **Fix**: Change it to `from src.bayes_decisions_model import ...`.
3. **Invalid Covariance Matrix Unpacking in GMM**:
   - **Issue**: [gaussian_mixture_models.py](file:///c:/Users/matti/Documents/PoliTO/Machine%20Learning%20and%20Pattern%20Recognition/Project/BiometricSpoofingClassification/src/gaussian_mixture_models.py#L199) calls `C, mu = computeCovariance(X)`. However, `computeCovariance(X)` returns only a single numpy array (`C`). This raises a `TypeError` at runtime. Furthermore, `computeMean` (which yields `mu`) is not imported.
   - **Fix**: Import `computeMean` from `src.utils`, and rewrite:
     ```python
     C = computeCovariance(X)
     mu = computeMean(X)
     ```
4. **PCA Covariance Indexing Bug**:
   - **Issue**: In [dimensionality_reduction.py](file:///c:/Users/matti/Documents/PoliTO/Machine%20Learning%20and%20Pattern%20Recognition/Project/BiometricSpoofingClassification/src/dimensionality_reduction.py#L44), PCA trains with `C = computeCovariance(D)[0]`. Because `computeCovariance(D)` is a 2D matrix, `[0]` extracts only the first row. This causes `np.linalg.eigh(C)` to fail with a `LinAlgError` since the input is 1D.
   - **Fix**: Remove `[0]` to assign the entire 2D covariance matrix to `C`.
5. **No Return Value in `computeCorrelationMatrix`**:
   - **Issue**: [utils.py](file:///c:/Users/matti/Documents/PoliTO/Machine%20Learning%20and%20Pattern%20Recognition/Project/BiometricSpoofingClassification/src/utils.py#L48-L49) computes the correlation matrix using the division operator `/` but does not assign it or return it.
   - **Fix**: Return the computed matrix (`return C / (...)`).
6. **Incorrect Argument Signature/Import in `visualization.py`**:
   - **Issue**: `visualization.py` calls `compute_optimal_bayes_decisions` on lines 135 and 140, but the function is not imported. Moreover, it passes three arguments (`effPrior, raw_scores, LVAL`), whereas the function only accepts two (`llr, t`).
   - **Fix**: Import the function, and verify/update the arguments passed to align with the definition in `bayes_decisions_model.py`.
7. **Attribute Error in `MultivariateGaussianClassifier.predict_multiclass`**:
   - **Issue**: On [gaussian_models.py](file:///c:/Users/matti/Documents/PoliTO/Machine%20Learning%20and%20Pattern%20Recognition/Project/BiometricSpoofingClassification/src/gaussian_models.py#L108), `loglikelihoods` expects arrays for `means` and `covariances`, but is passed the dictionary objects (`self.means`, `self.covariances`).
   - **Fix**: Restructure `loglikelihoods` or convert the dictionary to a numpy array prior to calling.