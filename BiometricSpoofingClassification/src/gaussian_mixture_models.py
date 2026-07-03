# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import scipy
from src.utils import computeCovariance, vcol
from src.bayes_decisions_model import compute_predictions_with_llr
from src.gaussian_models import logpdf_GAU_ND

def logpdf_GMM(X, gmm):
    """
    Computes the log-density of a GMM for a set of samples contained in matrix X
    
    Args:
        X: Matrix of samples (D, N), where D is the number of features and N is the number of samples
        gmm: List of Gaussian components, where each component is a tuple (w, mu, C)
            w: Weight of the component
            mu: Mean of the component
            C: Covariance matrix of the component
    
    Returns:
        Vector of log-densities for each sample
    """
    M = len(gmm)
    S = np.zeros((M, X.shape[1])) # S shape (M, N). S[g, ;] is the log-density of the g-th Gaussian component for all samples in X
    for g in range(M):
        w, mu, C = gmm[g]
        S[g, :] = logpdf_GAU_ND(X, mu, C) + np.log(w)
    # We use the log-sum-exp trick to compute the log-density of the GMM
    logdens = scipy.special.logsumexp(S, axis=0)
    return logdens # Output is a vector (N,) where each element is the log-density of the GMM for the corresponding sample in X
    
def GMM_EM_estimation(X, gmm_init, psi=None, eps_ll=1e-6):
    """
    Estimated the parameters of a GMM that maximize the likelihood for a training set X
    through the Expectation-Maximization iterative algorithm.
    Stop when the log-likelihood increase is less that eps_ll.
    
    Args:
        X: Matrix of samples (D, N), where D is the number of features and N is the number of samples
        gmm_init: Initial GMM, where each component is a tuple (w, mu, C)
            w: Weight of the component
            mu: Mean of the component
            C: Covariance matrix of the component
        psi: Regularization parameter for the covariance matrix
        eps_ll: Stopping criterion for the EM algorithm (log-likelihood difference)
    
    Returns:
        Updated GMM with estimated parameters
    """
    M = len(gmm_init)
    gmm = gmm_init.copy()
    S = np.zeros((M, X.shape[1]))

    currentLoglikelihood = None

    while True:
        # Step 1 (E-Step): compute posterior prob. for each component of GMM for each sample
        for g in range(M):
            w, mu, C = gmm[g]
            S[g, :] = logpdf_GAU_ND(X, mu, C) + np.log(w) # Joint density of g-th component and X
            
        logSMarginal = scipy.special.logsumexp(S, axis=0) # Marginal densities
        newLoglikelihood = logSMarginal.mean()
        
        # Stopping criterion: we can check the log-likelihood of the data given the model parameters and stop when it does not increase significantly anymore
        if currentLoglikelihood is not None and np.abs(newLoglikelihood - currentLoglikelihood) < eps_ll:
            break
            
        currentLoglikelihood = newLoglikelihood
        
        logSPost = S - logSMarginal
        SPost = np.exp(logSPost)

        # Step 2 (M-Step): Update model parameters
        for g in range(M):
            Z_g = SPost[g, :].sum() # Normalization factor for g-th component
            F_g = vcol((X * SPost[g, :]).sum(axis=1)) # Weighted sum of samples for g-th component
            S_g = (X * SPost[g, :]) @ X.T # Weighted sum of squared samples for g-th component

            new_mu_g = F_g / Z_g
            new_C_g = S_g / Z_g - new_mu_g @ new_mu_g.T
            new_w_g = Z_g / X.shape[1]
            
            # Regularize covariance matrix if psi is not None
            if psi is not None:
                U, s, Vh = np.linalg.svd(new_C_g)
                s[s < psi] = psi
                new_C_g = U @ (vcol(s) * U.T)

            gmm[g] = (new_w_g, new_mu_g, new_C_g)

    return gmm

def LBG_split(gmm_in, alpha):
    """
    Split a Gaussian component into two components with mean shift alpha along the direction of maximum variance.
    Used in the LBG algorithm.
    
    Args:
        gmm_in: List of Gaussian components, where each component is a tuple (w, mu, C)
            w: Weight of the component
            mu: Mean of the component
            C: Covariance matrix of the component
        alpha: Shift factor along the direction of maximum variance
    
    Returns:
        Updated GMM with split components
    """
    gmm_out = []
    for w, mu, C in gmm_in:
        U, s, Vh = np.linalg.svd(C)
        d = U[:, 0:1] * s[0]**0.5 * alpha
        gmm_out.append((w/2, mu - d, C))
        gmm_out.append((w/2, mu + d, C))
    return gmm_out

class GaussianMixtureModel():
    def __init__(self):
        self.parameters_class_true = None
        self.parameters_class_false = None
    
    def train(self, DTR, LTR, numComponents, alpha=0.1, psi=0.01):
        """
        Trains the Gaussian Mixture Model (GMM) classifier on the training set DTR with labels LTR.
        
        Args:
            DTR: Training data (D, N)
            LTR: Training labels (N,)
            numComponents: Number of Gaussian components in the GMM
            alpha: Shift factor along the direction of maximum variance (LBG algorithm)
            psi: Regularization parameter for the covariance matrix
        """
        DTR0 = DTR[:, LTR == 0] # False samples
        DTR1 = DTR[:, LTR == 1] # True samples

        print("\nTraining GMM for class 0...")
        self.parameters_class_false = evaluate_GMM_parameters_LBG_EM(DTR0, numComponents, alpha, psi)
        print("\nTraining GMM for class 1...")
        self.parameters_class_true = evaluate_GMM_parameters_LBG_EM(DTR1, numComponents, alpha, psi)

    def get_scores(self, X):
        """
        Computes the Log-Likelihood Ratios (LLRs) for a set of samples X.
        
        Args:
            X: Data to classify (D, N)
            
        Returns:
            LLRs: Log-Likelihood Ratios (N,)
        """
        # Compute class-conditional log-densities (scores for each class)
        logS_false = logpdf_GMM(X, self.parameters_class_false)
        logS_true = logpdf_GMM(X, self.parameters_class_true)

        # Compute Log-Likelihood Ratios (LLRs)
        LLRs = logS_true - logS_false

        return LLRs

    def predict(self, X, t=0):
        """
        Predicts the class labels for a set of samples X using the GMM classifier.
        
        Args:
            X: Data to classify (D, N)
            t: Threshold for the classification
            
        Returns:
            PVAL: Predicted labels (N,)
        """
        LLRs = self.get_scores(X)

        # Compute predictions
        PVAL = compute_predictions_with_llr(LLRs, t)

        return PVAL
        
def evaluate_GMM_parameters_LBG_EM(X, numComponents, alpha=0.1, psi=0.01):
    """
    Evaluates the parameters of a GMM for a training set X using the LBG + EM algorithm for samples of one class.

    Algorithm:
    1. Compute the best single Gaussian model for the training set X
    2. Initialize GMM with one component (the best single Gaussian model)
    3. Split each component into two components with mean shift alpha along the direction of maximum variance
    4. Estimate the optimal parameters of the GMM using EM algorithm after splitting the components
    5. Repeat steps 3 and 4 until numComponents components are obtained
    
    Args:
        X: Matrix of samples (D, N), where D is the number of features and N is the number of samples
        numComponents: Number of Gaussian components in the GMM
        alpha: Shift factor along the direction of maximum variance (LBG algorithm)
        psi: Regularization parameter for the covariance matrix
    
    Returns:
        GMM with estimated parameters
    """
    # Compute best single Gaussian model
    C, mu = computeCovariance(X)
    
    # Regularize covariance matrix if psi is not None
    if psi is not None:
        U, s, Vh = np.linalg.svd(C)
        s[s < psi] = psi
        C = U @ (vcol(s) * U.T)
        
    # Estimate GMM parameters [(w_j, mu_j, C_j)] for each class using LBG + EM
    gmm = [(1.0, mu, C)]                            # Initialize GMM with one component
    for _ in range(int(np.log2(numComponents))):    # Repeat the split for log2(numComponents) times in order to obtain numComponents components
        print(f"\nProgress: {(_ + 1) / int(np.log2(numComponents)) * 100:.1f}%", end='\r')
        gmm = LBG_split(gmm, alpha)                 # Split each component into two components with mean shift alpha along the direction of maximum variance
        gmm = GMM_EM_estimation(X, gmm, psi=psi)    # Estimate the optimal parameters of the GMM using EM algorithm after splitting the components
    print("\nProgress: 100.0%\n")
    return gmm