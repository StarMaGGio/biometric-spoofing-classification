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
(Iris Dataset Lab)

---
### 2. Dimensionality Reduction

We explore two dimensionality reduction techniques to project the 6-dimensional feature space into lower dimensions: **Principal Component Analysis (PCA)** and **Linear Discriminant Analysis (LDA)**.

#### 🔹 Principal Component Analysis (PCA)
PCA is an unsupervised technique that projects the samples onto the directions of maximum variance to retain as much information as possible.
* **2D Feature Projection:** Below is the scatter plot comparison showing the original first two features vs. the features projected onto the first two principal components.

  | Original Features (2D space) | PCA Transformed Features (2D space) |
  | :---: | :---: |
  | <img src="BiometricSpoofingClassification/images/Fea1_2_scatterPlot.png" width="400"> | <img src="BiometricSpoofingClassification/images/PCA1_2_scatter_plot.png" width="400"> |

* **Analysis of PCA Directions:** Below are the histograms of the dataset projected onto each of the 6 PCA directions (sorted by descending variance):

  | PCA Direction 1 | PCA Direction 2 | PCA Direction 3 |
  | :---: | :---: | :---: |
  | <img src="BiometricSpoofingClassification/images/PCA_1.png" width="400"> | <img src="BiometricSpoofingClassification/images/PCA_2.png" width="400">  | <img src="BiometricSpoofingClassification/images/PCA_3.png" width="400"> |
  | **PCA Direction 4** | **PCA Direction 5** | **PCA Direction 6** |
  | <img src="BiometricSpoofingClassification/images/PCA_4.png" width="400"> | <img src="BiometricSpoofingClassification/images/PCA_5.png" width="400"> | <img src="BiometricSpoofingClassification/images/PCA_6.png" width="400"> |

---

#### 🔸 Linear Discriminant Analysis (LDA)
Unlike PCA, LDA is a supervised technique that finds the projection subspace that maximizes class separability by maximizing the ratio between the between-class variance ($S_B$) and within-class variance ($S_W$).

  | $S_B$ vs $S_W$ Covariance | Classification Threshold |
  | :---: | :---: |
  | <img src="BiometricSpoofingClassification/images/SBvsSW_sketch.png" width="400"> | <img src="BiometricSpoofingClassification/images/PCA_classification_sketch.png" width="400"> |

* **1D LDA Projection:** Projected onto the 1-dimensional LDA subspace, the samples of the two classes exhibit significant separation:

  | LDA Projection |
  | :---: |
  | <img src="BiometricSpoofingClassification/images/LDA_1.png" width="400"> |

* **Classification Performance:** Using the average of the projected class means as the decision threshold yields the following result:
  ```
  Number of LDA directions: 1
  Threshold: -0.01853
  LDA-only error rate: 0.09300 (9.30%)
  ```
---

#### 🔄 Joint PCA + LDA Classification
To reduce noise, we project data via PCA before applying LDA.

  | Performance |
  | :---: |
  | <img src="BiometricSpoofingClassification/images/PCA+LDA_error_rate.png" alt="Performances" width="400"> |

---
### 3. Generative Gaussian Models

  In **Generative Models** we start from the assumption that observed samples, conditioned to model parameters, are *indipendent and identically distributed*.<br>
  In the case of **Multivariate Gaussian Classifiers** we assume that each feature can be described by a *Gaussian probability distribution*.

  The training of the model consists in fact in the *maximization* of the **likelihood function** of the data, that is the Joint-Density, obtained as the product (sum) of the likelihoods (log-likelihoods).

  $$ \mathcal{L}(\mu, \Sigma; X) = \sum_{i=1}^N \log p(x_i | \mu, \Sigma) $$

  | Generative Model Assumption | Maximum Likelihood Approach |
  | :---: | :---: |
  | <img src="BiometricSpoofingClassification/images/GenModelAssumption.png" alt="GenAssumption" width="400"> | <img src="BiometricSpoofingClassification/images/MaximumLikelihood.png" alt="MaxLikelihood" width="400"> |

  Multivariate Generative Classifiers are divided in three main variants, based on their assumptions on the dataset.
  - As we said **Multivariate Gaussian Classifiers** assume *indipendent and identically distributed* data.
  - **Naive Bayes Classifiers** also assume that inside the same class, single components of the features vector are approximately indipendent.
  - **Tied Gaussian Classifiers** add instead the assumption that the different classes share the same covariance matrix.
  These different assumptions on data distributions affect the shape of the estimated probability density functions and the resulting decision boundary between the different models.

  | Estimated Densities Differences | Covariances and Decision Boundary Differences |
  | :---: | :---: |
  | <img src="BiometricSpoofingClassification/images/EstDensDiff.png" alt="EstDensDiff" width="400"> | <img src="BiometricSpoofingClassification/images/CovDecDiff.png" alt="CovDecDiff" width="400"> |

#### 🔹 Multivariate Gaussian Classifier (MVG)
* **Classification Performance:**
  ```
  Threshold: 0
  MVG error rate - features 1 to 6: 0.07000
  ```
* **Analysis of estimated probability densities:**
  | | Feature 1 | Feature 2 | Feature 3 | Feature 4 | Feature 5 | Feature 6 |
  | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
  | **Class 0** | ![MVG 1](BiometricSpoofingClassification/images/Gau_Distr_of_Fea_1_Class_0_Model_MVG.png) | ![MVG 2](BiometricSpoofingClassification/images/Gau_Distr_of_Fea_2_Class_0_Model_MVG.png) | ![MVG 3](BiometricSpoofingClassification/images/Gau_Distr_of_Fea_3_Class_0_Model_MVG.png) | ![MVG 4](BiometricSpoofingClassification/images/Gau_Distr_of_Fea_4_Class_0_Model_MVG.png) | ![MVG 5](BiometricSpoofingClassification/images/Gau_Distr_of_Fea_5_Class_0_Model_MVG.png) | ![MVG 6](BiometricSpoofingClassification/images/Gau_Distr_of_Fea_6_Class_0_Model_MVG.png) |
  | **Class 1** | ![MVG 7](BiometricSpoofingClassification/images/Gau_Distr_of_Fea_1_Class_1_Model_MVG.png) | ![MVG 8](BiometricSpoofingClassification/images/Gau_Distr_of_Fea_2_Class_1_Model_MVG.png) | ![MVG 9](BiometricSpoofingClassification/images/Gau_Distr_of_Fea_3_Class_1_Model_MVG.png) | ![MVG 10](BiometricSpoofingClassification/images/Gau_Distr_of_Fea_4_Class_1_Model_MVG.png) | ![MVG 11](BiometricSpoofingClassification/images/Gau_Distr_of_Fea_5_Class_1_Model_MVG.png) | ![MVG 12](BiometricSpoofingClassification/images/Gau_Distr_of_Fea_6_Class_1_Model_MVG.png) |

* **Bivariate Gaussian Probability Densities:**
  | Feature 1-2 | Feature 3-4 | Feature 4-5 |
  | :---: | :---: | :---: |
  |![MVG 1](BiometricSpoofingClassification/images/MVG_2D_Density_Ellipses_Fea1_Fea2.png) | ![MVG 2](BiometricSpoofingClassification/images/MVG_2D_Density_Ellipses_Fea3_Fea4.png) | ![MVG 3](BiometricSpoofingClassification/images/MVG_2D_Density_Ellipses_Fea5_Fea6.png) |

---
#### 🔹 Naive Bayes Gaussian Classifier
* **Classification Performance:**
  ```
  Threshold: 0
  Naive Bayes Gaussian error rate - features 1 to 6: 0.07200
  ```

* **Bivariate Gaussian Probability Densities:**
  | Feature 1-2 | Feature 3-4 | Feature 4-5 |
  | :---: | :---: | :---: |
  |![NBG 1](BiometricSpoofingClassification/images/NBG_2D_Density_Ellipses_Fea1_Fea2.png) | ![NBG 2](BiometricSpoofingClassification/images/NBG_2D_Density_Ellipses_Fea3_Fea4.png) | ![NBG 3](BiometricSpoofingClassification/images/NBG_2D_Density_Ellipses_Fea5_Fea6.png) |

---
#### 🔹 Tied Gaussian Classifier
* **Classification Performance:**
  ```
  Threshold: 0
  Tied Gaussian error rate - features 1 to 6: 0.09300
  ```
* **Analysis of estimated probability densities:**
  | | Feature 1 | Feature 2 | Feature 3 | Feature 4 | Feature 5 | Feature 6 |
  | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
  | **Class 0** | ![TG 1](BiometricSpoofingClassification/images/Gau_Distr_of_Fea_1_Class_0_Model_TG.png) | ![TG 2](BiometricSpoofingClassification/images/Gau_Distr_of_Fea_2_Class_0_Model_TG.png) | ![TG 3](BiometricSpoofingClassification/images/Gau_Distr_of_Fea_3_Class_0_Model_TG.png) | ![TG 4](BiometricSpoofingClassification/images/Gau_Distr_of_Fea_4_Class_0_Model_TG.png) | ![TG 5](BiometricSpoofingClassification/images/Gau_Distr_of_Fea_5_Class_0_Model_TG.png) | ![TG 6](BiometricSpoofingClassification/images/Gau_Distr_of_Fea_6_Class_0_Model_TG.png) |
  | **Class 1** | ![TG 7](BiometricSpoofingClassification/images/Gau_Distr_of_Fea_1_Class_1_Model_TG.png) | ![TG 8](BiometricSpoofingClassification/images/Gau_Distr_of_Fea_2_Class_1_Model_TG.png) | ![TG 9](BiometricSpoofingClassification/images/Gau_Distr_of_Fea_3_Class_1_Model_TG.png) | ![TG 10](BiometricSpoofingClassification/images/Gau_Distr_of_Fea_4_Class_1_Model_TG.png) | ![TG 11](BiometricSpoofingClassification/images/Gau_Distr_of_Fea_5_Class_1_Model_TG.png) | ![TG 12](BiometricSpoofingClassification/images/Gau_Distr_of_Fea_6_Class_1_Model_TG.png) |
* **Bivariate Gaussian Probability Densities:**
  | Feature 1-2 | Feature 3-4 | Feature 4-5 |
  | :---: | :---: | :---: |
  |![TG 1](BiometricSpoofingClassification/images/TG_2D_Density_Ellipses_Fea1_Fea2.png) | ![TG 2](BiometricSpoofingClassification/images/TG_2D_Density_Ellipses_Fea3_Fea4.png) | ![TG 3](BiometricSpoofingClassification/images/TG_2D_Density_Ellipses_Fea5_Fea6.png) |

* **Theorical Equivalence Between Tied Gaussian Classifier and LDA**

  During the performance analysis, a noteworthy result emerged: the error rate obtained using the Tied Gaussian model is numerically identical to the error rate obtained using Linear Discriminant Analysis (LDA) and classifying samples with respect to the mean of the projected means ($\frac{m_0 + m_1}{2}$).<br><br>
  This empirical result is perfectly justified by the mathematical theory of the two classifiers. Both methods assume (implicitly or explicitly) that the two classes share the same intra-class dispersion/covariance matrix ($\Sigma_W$). In the case of the Tied Gaussian classifier, sharing the covariance matrix causes the quadratic discriminant functions to collapse into linear functions. The Log-Likelihood Ratio (LLR) development generates a weight vector $w = \Sigma_W^{-1}(\mu_1 - \mu_0)$, which is exactly the same projection direction identified by maximizing the Fisher criterion in the LDA algorithm.<br><br>
  Consequently, the decision boundary computed by the generative model (Tied Gaussian) exactly coincides with the separating hyperplane computed by the projective model (LDA with intermediate threshold). The identity of the performances therefore confirms the correct implementation and the solid theoretical link between the two approaches in the case of binary classification.

---
#### 🔹 Covariance and Correlation Matrices Analysis
  | | Class 0 | Class 1 |
  | :---: | :---: | :---: |
  | Covariance Matrix | <img src="BiometricSpoofingClassification/images/CovMatrClass0.png" alt="CM 0" width="400"> | <img src="BiometricSpoofingClassification/images/CovMatrClass1.png" alt="CM 1" width="400"> |
  | Correlation Matrix | <img src="BiometricSpoofingClassification/images/CorrMatrClass0.png" alt="CM 0" width="400"> | <img src="BiometricSpoofingClassification/images/CorrMatrClass1.png" alt="CM 1" width="400"> |

* **Explanation of Naive Bayes Model Performances**

  As we can see, the six features of these fingerprints are inherently uncorrelated. They are statistically independent.

  - MVG uses these tiny values ​​(e.g., 0.026) to tilt its ellipses by a fraction of a millimeter invisible to the naked eye.

  - Naive Bayes makes its typical "naive assumption," setting these values ​​to zero. But since in reality they were already close to zero, the Naive Bayes approximation turns out to be incredibly accurate!

  This also explain why Naive Bayes has likely the same performances as the MVG

* **Explaination of Decreasing Performances of Tied Gaussian Model**

  Looking at the two covariance matrices on the main diagonal (the variances):

  - Index 1 (Feature 2): In Class 0, the variance is very high (1.421). In Class 1, it drops to 0.578.

  - Index 0 (Feature 1): In Class 0, the variance is 0.570. In Class 1, it skyrockets to 1.430.

  This is a huge amount of discriminative information for a classifier!

  The Tied Covariance model takes the 1.421 and 0.578, averages them (about 1.0), and imposes it on both classes. This prevents this geometric difference from being used to distinguish the two classes. This is why the error increases.

---
#### 🔹 Performances Analysis with Reduced Features Set

  |  | All features | Features 1-2-3-4 | Features 1-2 | Features 3-4 |
  | :---: | :---: | :---: | :---: | :---: |
  | **MVG** | error rate: 0.07000 | error rate: 0.07950 | error rate: 0.36500 | error rate: 0.09450 |
  | **Naive Bayes** | error rate: 0.07200 | error rate: 0.07650 | error rate: 0.36300 | error rate: 0.09450 |
  | **Tied Gaussian** | error rate: 0.09300 | error rate: 0.09500 | error rate: 0.49450 | error rate: 0.09400 |

  The table highlights how discriminatory power is not uniformly distributed across the dataset's features. The set restricted to Features 3-4 yields a much lower error rate (~9.4%) than Features 1-2 (>36%), demonstrating that the latter contain little information for class separation when considered individually.

  Analyzing the models' behavior in relation to their mathematical assumptions, three key dynamics emerge:

  1. Robustness of the Naive Assumption: The Naive Bayes classifier maintains performance nearly comparable to the full MVG in all configurations. On the Features 1-2-3-4 set, the strong independence assumption even acts as a regularizer, allowing Naive Bayes to slightly outperform the MVG, mitigating overfitting on spurious correlations.

  2. Sensitivity of the Tied Gaussian to Class-Specific Variances: The steep decline in performance of the Tied Gaussian model on Features 1-2 (Error Rate 0.49450, close to random guessing) is a direct consequence of its founding assumption. As highlighted by the preliminary analysis, Feature 2 exhibits markedly different variances between the Fake and Genuine classes. Imposing a shared global covariance (Tied) suppresses this divergence, destroying the feature's discriminative utility.

  3. Tied Validity on Features 3-4: Conversely, the Tied Gaussian proves to be the marginally better model when evaluating only Features 3-4. This suggests that for this specific subset, intra-class dispersions are natively homogeneous, making the Tied covariance assumption not only valid but beneficial for the statistical robustness of the estimate.

---
#### 🔹 Effects of PCA as Preprocessing Technique on Gaussian Models
  
  | | PCA Effects on MVG | PCA Effects on Naive Bayes | PCA Effects on Tied Gaussian |
  | :---: | :---: | :---: | :---: |
  | **Performances** | ![Performance1](BiometricSpoofingClassification/images/FunctionMVGPreprocessing.png) | ![Performance2](BiometricSpoofingClassification/images/FunctionNBGPreprocessing.png) | ![Performance3](BiometricSpoofingClassification/images/FunctionTGPreprocessing.png) |

  Performance analysis as the number of PCA components ($m$) varies highlights distinct behaviors closely related to the mathematical assumptions of the three models:<br>
  - **MVG**: The Error Rate decreases monotonically as the PCA directions increase, reaching a minimum at $m=6$. This indicates that the discriminative information is distributed transversally across the feature space. Since MVG is invariant to linear transformations, for $m=6$ (pure rotation) the model exactly restores the performance obtained on the raw data.
  - **Naive Bayes**: A counterintuitive but theoretically sound phenomenon is observed: applying PCA degrades performance compared to the original data (the error at $m=6$ stands at ~0.089 versus the starting 0.072). Since PCA decorrelates the dataset globally but not necessarily the individual class-conditional distributions, space rotation introduces previously absent intra-class covariances.The Naive Bayes classifier, ignoring these covariances due to its strong assumption, suffers a degradation in predictive performance, demonstrating its non-invariance to rotations.
  - **Tied Gaussian**: The curve is invariant with respect to the dimensionality of the PCA space (error stable at ~0.093). This trend confirms that the model's performance limitation lies not in the "curse of dimensionality" or the variance of noise, but rather in the strict assumption of sharing the global covariance matrix, which suppresses the valuable dispersion differences between the Fake and Genuine classes.

---

### 4. Model Evaluation - Bayes Risk

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
- [ ] **Visualization Enhancements**:
  - *Task*: Generalize and write better existing visualization functions
  - *Task*: Write visualization function for maximum likelihood estimation for GMM. Plot of density function over the normalized histogram of features
  - *Task*: Write a generalized visualization function to plot on 2D/3D graph data points and decision boundaries for a given classifier