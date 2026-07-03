# pyrefly: ignore [missing-import]
from typing import override
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import scipy
from src.utils import vcol, vrow

class SupportVectorMachine():
    def __init__(self):
        self.w = None
        self.b = None

    def train(self, DTR, LTR, C, K):
        """
        Function to train the model using Support Vector Machine
        

        Parameters
        ----------
        DTR : (numpy.ndarray)
            Training Features matrix of shape (n_features, n_samples).
        LTR : (numpy.ndarray)
            Training Labels vector of shape (n_samples,).
        C : (float)
            Regularization parameter.
        K : (float)
            Bias parameter.

        Returns
        -------
        None
        """
        DTR_EXT = np.vstack([DTR, np.ones((1, DTR.shape[1])) * K]) # Append a row of elements all = K to incorporate the bias
        ZTR = LTR * 2.0 - 1.0 # Map labels to -1/+1
        H = np.dot(DTR_EXT.T, DTR_EXT) * vcol(ZTR) * vrow(ZTR) # H_i,j = z_i*z_j*x_i^T*x_j where x = DTR_EXT and z = ZTR

        # Dual objective and gradient
        def fOpt(alpha):
            Halpha = H @ vcol(alpha)
            loss = 0.5 * (vrow(alpha) @ Halpha).ravel() - alpha.sum() # L^D(alpha) = -J^D(alpha)
            grad = Halpha.ravel() - np.ones(alpha.size)
            return loss, grad

        # Search the minimazer of the loss function to compute the optimal dual solution
        alphaStar, _, _ = scipy.optimize.fmin_l_bfgs_b(fOpt, np.zeros(DTR_EXT.shape[1]), bounds = [(0, C) for i in LTR], factr=np.nan, pgtol=1e-5)
    
        # Function to compute primal objective to check duality gap
        def primalLoss(w_hat):
            S = (vrow(w_hat) @ DTR_EXT).ravel() # Scores
            return 0.5 * np.linalg.norm(w_hat)**2 + C * np.maximum(0, 1 - ZTR * S).sum()
    
        # Compute primal solution from the dual solution
        w_hat = (vrow(alphaStar) * vrow(ZTR) * DTR_EXT).sum(1)

        # Compute and set optimal weights vector and bias
        self.w = w_hat[0:DTR.shape[0]]
        self.b = w_hat[-1] * K # b must be rescaled in case K != 1, since we want to compute w^Tx + b * K
        
        # Compute and print the duality gap to verify the solution is correct
        primalLoss, dualLoss = primalLoss(w_hat), -fOpt(alphaStar)[0]
        print('SVM - K %f - C %f - primal loss %e - dual loss %e - duality gap %e' % (K, C, primalLoss, dualLoss[0], primalLoss - dualLoss[0]))

    def get_scores(self, X):
        # Compute scores
        SVAL = (vrow(self.w) @ X + self.b).ravel() 
        return SVAL

    def predict(self, X):
        # Compute predictions            
        PVAL = (self.get_scores(X) > 0) * 1
        return PVAL

class KernelSupportVectorMachine(SupportVectorMachine):
    def __init__(self):
        super().__init__()
        self.fScore = None

    def train(self, DTR, LTR, C, kernelFunc, eps=1.0):
        """
        Function to train the model using kernel support vector machine
        
        Parameters
        ----------
        DTR : (numpy.ndarray)
            Training Features matrix of shape (n_features, n_samples).
        LTR : (numpy.ndarray)
            Training Labels vector of shape (n_samples,).
        C : (float)
            Regularization parameter.
        K : (float)
            Bias parameter.
        kernelFunc : (function)
            Kernel function.
        eps : (float)
            Epsilon parameter.
        
        Returns
        -------
        None
        """
        ZTR = LTR * 2.0 - 1.0 # Convert labels to -1/+1
        K = kernelFunc(DTR, DTR) + eps # Replace DTR dot product with Kernel Function
        H = np.outer(ZTR, ZTR) * K
        
        # Function to compute Dual objective and gradient
        def fOpt(alpha):
            Halpha = H @ vcol(alpha)
            loss = 0.5 * (vrow(alpha) @ Halpha).ravel() - alpha.sum() # L^D(alpha) = -J^D(alpha)
            grad = Halpha.ravel() - np.ones(alpha.size)
            return loss, grad
        
        # Search the minimazer of the loss function
        alphaStar, _, _ = scipy.optimize.fmin_l_bfgs_b(fOpt, np.zeros(DTR.shape[1]), bounds = [(0, C) for i in LTR], factr=np.nan, pgtol=1e-5)
        
        # Function to compute primal loss
        def primalLoss(alpha):
            Halpha = H @ vcol(alpha)
            return 0.5 * (vrow(alpha) @ Halpha) + C * np.maximum(0, 1 - Halpha).sum()

        # Compute and print the duality gap
        primalLoss, dualLoss = primalLoss(alphaStar), -fOpt(alphaStar)[0][0]
        print('SVM (Kernel) - C %f - primal loss %e - dual loss %e - duality gap %e' % (C, primalLoss, dualLoss, primalLoss - dualLoss))
        
        # Function to compute scores for incoming samples
        def fScore(DTE):
            K = kernelFunc(DTR, DTE) + eps
            H = vcol(alphaStar) * vcol(ZTR) * K
            return H.sum(0)

        self.fScore = fScore

    @override
    def get_scores(self, X):
        SVAL = self.fScore(X)
        return SVAL

    @override
    def predict(self, X):
        return super().predict(X)

    