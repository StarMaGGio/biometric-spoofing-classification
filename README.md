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

## Project Analysis and Observations
### 1. Dataset Analysis

### 2. Dimensionality Reduction

We explore two dimensionality reduction techniques to project the 6-dimensional feature space into lower dimensions: **Principal Component Analysis (PCA)** and **Linear Discriminant Analysis (LDA)**.

#### 🔹 Principal Component Analysis (PCA)
PCA is an unsupervised technique that projects the samples onto the directions of maximum variance to retain as much information as possible.
* **2D Feature Projection:** Below is the scatter plot comparison showing the original first two features vs. the features projected onto the first two principal components.

| Original Features (2D space) | PCA Transformed Features (2D space) |
| :---: | :---: |
| ![Original Features](BiometricSpoofingClassification/images/Fea1_2_scatterPlot.png) | ![PCA Transformed Features](BiometricSpoofingClassification/images/PCA1_2_scatter_plot.png) |

* **Analysis of PCA Directions:** Below are the histograms of the dataset projected onto each of the 6 PCA directions (sorted by descending variance):

| PCA Direction 1 | PCA Direction 2 | PCA Direction 3 |
| :---: | :---: | :---: |
| ![PCA 1](BiometricSpoofingClassification/images/PCA_1.png) | ![PCA 2](BiometricSpoofingClassification/images/PCA_2.png) | ![PCA 3](BiometricSpoofingClassification/images/PCA_3.png) |
| **PCA Direction 4** | **PCA Direction 5** | **PCA Direction 6** |
| ![PCA 4](BiometricSpoofingClassification/images/PCA_4.png) | ![PCA 5](BiometricSpoofingClassification/images/PCA_5.png) | ![PCA 6](BiometricSpoofingClassification/images/PCA_6.png) |

---

#### 🔸 Linear Discriminant Analysis (LDA)
Unlike PCA, LDA is a supervised technique that finds the projection subspace that maximizes class separability by maximizing the ratio between the between-class variance ($S_B$) and within-class variance ($S_W$).

| $S_B$ vs $S_W$ Covariance Sketch | Classification Threshold Sketch |
| :---: | :---: |
| ![LDA Sketch](BiometricSpoofingClassification/images/SBvsSW_sketch.png) | ![Threshold Sketch](BiometricSpoofingClassification/images/PCA_classification_sketch.png) |

* **1D LDA Projection:** Projected onto the 1-dimensional LDA subspace, the samples of the two classes exhibit significant separation:

| LDA Projection |
| :---: |
| ![LDA1](BiometricSpoofingClassification/images/LDA_1.png) |

* **Classification Performance:** Using the average of the projected class means as the decision threshold yields the following result:
  ```
  Number of LDA directions: 1
  Threshold: -0.01853
  LDA-only error rate: 0.09300 (9.30%)
  ```
LDA maximizes class separability using the ratio of between-class ($S_B$) and within-class ($S_W$) variances.
* **Table 3: LDA Model characteristics.**

| Concept | Visual Representation |
| :---: | :---: |
| **Covariance Modeling** | ![Sketch](BiometricSpoofingClassification/images/SBvsSW_sketch.png) |
| **Decision Threshold** | ![Threshold](BiometricSpoofingClassification/images/PCA_classification_sketch.png) |

---

#### 🔄 Joint PCA + LDA Classification
To reduce noise, we project data via PCA before applying LDA.
* **Error Analysis:**
  - ![Performance](BiometricSpoofingClassification/images/PCA+LDA_error_rate.png)

---

## 📝 TODOs & Recommended Modifications

Below is a consolidated status list of project tasks and recommended code fixes identified during design and validation.

### 🔍 Code TODOs (from `main.py`)
- [ ] **Pipeline Modularity** (Line 19):
  - *Task*: Move all the pipeline analysis functions defined in [main.py](file:///c:/Users/matti/Documents/PoliTO/Machine%20Learning%20and%20Pattern%20Recognition/Project/BiometricSpoofingClassification/main.py) (e.g., `PCA_LDA_effects_and_classification_analysis`, `compare_gaussian_models`, etc.) to separate files to keep the main script clean.
- [ ] **Logistic Regression Enhancements** (Line 239):
  - *Task*: Add reduced dataset analysis and quadratic feature expansion options to [analyze_logistic_regression_with_different_lambdas](file:///c:/Users/matti/Documents/PoliTO/Machine%20Learning%20and%20Pattern%20Recognition/Project/BiometricSpoofingClassification/main.py#L237) function.
- [ ] **Cross-Validation Function for Calibration** (Line 508):
  - *Task*: Refactor the K-Fold score calibration loop into a reusable function in a new `cross_validation.py` file.
- [ ] **Generalize DCF Computations** (Line 525):
  - *Task*: Generalize the loop computing actDCF/minDCF over different effective priors, package it into a utility function, and move it to a source module.
- [ ] **Generalize Bayes Error Plotting** (Line 551):
  - *Task*: Refactor the inline plotting logic for Bayes error curve into a function in `visualization.py`.
- [ ] **Cross-Validation Function for Fusion** (Line 599):
  - *Task*: Extract the K-Fold fusion calibration loop into the `cross_validation.py` module.

### 🛠️ Recommended Code & Design Fixes

- [ ] **1. Undefined Function `compute_predictions_with_llr`**:
  - *Issue*: Imported in multiple modules (e.g., [gaussian_models.py](file:///c:/Users/matti/Documents/PoliTO/Machine%20Learning%20and%20Pattern%20Recognition/Project/BiometricSpoofingClassification/src/gaussian_models.py#L8)), but never defined inside `src/bayes_decisions_model.py`.
  - *Fix*: Re-route imports/calls to [compute_optimal_bayes_decisions](file:///c:/Users/matti/Documents/PoliTO/Machine%20Learning%20and%20Pattern%20Recognition/Project/BiometricSpoofingClassification/src/bayes_decisions_model.py#L74) or define the missing helper.
- [x] **2. Incorrect Module Path in Imports** (Fixed):
  - *Issue*: Models imported modules using prefix paths starting with `Project` (e.g., `from Project.src...`), which caused `ImportError` when run within the `BiometricSpoofingClassification` root directory.
  - *Fix*: Standardized all internal module imports to root from `src.models`.
- [ ] **3. Invalid Covariance Matrix Unpacking in GMM**:
  - *Issue*: [gaussian_mixture_models.py](file:///c:/Users/matti/Documents/PoliTO/Machine%20Learning%20and%20Pattern%20Recognition/Project/BiometricSpoofingClassification/src/gaussian_mixture_models.py#L199) calls `C, mu = computeCovariance(X)`, but `computeCovariance(X)` only returns `C`.
  - *Fix*: Import `computeMean` from `src.utils` and unpack properly.
- [x] **4. PCA Covariance Indexing Bug** (Fixed):
  - *Issue*: In [dimensionality_reduction.py](file:///c:/Users/matti/Documents/PoliTO/Machine%20Learning%20and%20Pattern%20Recognition/Project/BiometricSpoofingClassification/src/models/dimensionality_reduction.py#L44), PCA trained with `C = computeCovariance(D)[0]`, which extracted only the first row of the covariance matrix and caused `np.linalg.eigh` to crash.
  - *Fix*: Removed `[0]` to correctly assign the full 2D covariance matrix.
- [ ] **5. No Return Value in `computeCorrelationMatrix`**:
  - *Issue*: [utils.py](file:///c:/Users/matti/Documents/PoliTO/Machine%20Learning%20and%20Pattern%20Recognition/Project/BiometricSpoofingClassification/src/utils.py#L48-L49) computes the correlation matrix but does not return it.
  - *Fix*: Return the computed matrix (`return C / (...)`).
- [ ] **6. Incorrect Argument Signature/Import in `visualization.py`**:
  - *Issue*: `visualization.py` calls `compute_optimal_bayes_decisions` but passes three arguments whereas the function only accepts two.
  - *Fix*: Correct the argument signature and ensure proper import.
- [ ] **7. Attribute Error in `MultivariateGaussianClassifier.predict_multiclass`**:
  - *Issue*: Passes dictionary objects instead of numpy arrays to `loglikelihoods`.
  - *Fix*: Extract arrays from dictionaries before passing.
- [x] **8. LDA Sign Orientation / Class Positioning** (Fixed):
  - *Issue*: The arbitrary sign of the LDA projection vector could cause True/Genuine samples to be projected to the left (smaller values), failing the threshold classifier.
  - *Fix*: Standardized the training logic in [dimensionality_reduction.py](file:///c:/Users/matti/Documents/PoliTO/Machine%20Learning%20and%20Pattern%20Recognition/Project/BiometricSpoofingClassification/src/models/dimensionality_reduction.py#L128-L131) to guarantee that the projected mean of Class 1 is always greater than Class 0.