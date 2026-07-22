# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import math
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt

from src.models.utils import split_db_2to1, compute_effective_prior, compute_confusion_matrix
from src.models.bayes_decisions_model import compute_minimum_DCF, compute_actual_DCF
from src.models.visualization import plot_Bayes_error
from src.models.evaluation import compute_acc_err
from src.models.gaussian_models import MultivariateGaussianClassifier, NaiveBayesGaussianClassifier, TiedGaussianClassifier

def compare_effPriors_and_DCFs_for_different_applications(D, L):
    # -----------------------
    #  Evaluation/Bayes Risk
    # -----------------------
    
    # Divide the dataset in training and validation sets
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)

    inner_menu_option = int(input('\n Choose a model to evaluate:\n'
                                  '1. MVG\n'
                                  '2. Tied Gaussian\n'
                                  '3. Naive Bayes Gaussian\n'
                                  '4. Bayes Error Plot (Different Applications/Effective Priors)\n'
                                  '0. Back\n'))
    model = ""

    if inner_menu_option == 0: return

    if inner_menu_option != 4:
        pi = float(input("Prior probability of genuine sample: ").replace(',', '.'))
        Cfn = float(input("Cost of false negative: ").replace(',', '.'))
        Cfp = float(input("Cost of false positive: ").replace(',', '.'))
        
        # Compute effective prior
        eff_prior = compute_effective_prior(pi, Cfn, Cfp)
        print(f"Application (pi={pi}, Cfn={Cfn}, Cfp={Cfp}) -> Effective Prior: {eff_prior:.2f}")
        t = np.log((1-eff_prior)/eff_prior)

    # Compute optimal Bayes decisions on the validation set for the selected model
    LLRs = None
    match inner_menu_option:
        case 1:
            model = "MVG"
            MVG = MultivariateGaussianClassifier()
            MVG.train(DTR, LTR)
            LLRs = MVG.get_log_likelihood_ratios(DVAL)
            PVAL = MVG.predict_binary(DVAL, t)
        case 2:
            model = "Tied Gaussian"
            TG = TiedGaussianClassifier()
            TG.train(DTR, LTR)
            LLRs = TG.get_log_likelihood_ratios(DVAL)
            PVAL = TG.predict_binary(DVAL, t)
        case 3:
            model = "Naive Bayes Gaussian"
            NBG = NaiveBayesGaussianClassifier()
            NBG.train(DTR, LTR)
            LLRs = NBG.get_log_likelihood_ratios(DVAL)
            PVAL = NBG.predict_binary(DVAL, t)
        case 4:
            LLRs = np.zeros(shape=LVAL.shape)
            select_model = int(input("Select model: 1. MVG, 2. Tied Gaussian, 3. Naive Bayes Gaussian"))
            if select_model == 1:
                model = "MVG"
                MVG = MultivariateGaussianClassifier()
                MVG.train(DTR, LTR)
                LLRs = MVG.get_log_likelihood_ratios(DVAL)
            elif select_model == 2:
                model = "Tied Gaussian"
                TG = TiedGaussianClassifier()
                TG.train(DTR, LTR)
                LLRs = TG.get_log_likelihood_ratios(DVAL)
            elif select_model == 3:
                model = "Naive Bayes Gaussian"
                NBG = NaiveBayesGaussianClassifier()
                NBG.train(DTR, LTR)
                LLRs = NBG.get_log_likelihood_ratios(DVAL)
            plot_Bayes_error(LLRs, LVAL, model)
            return

    # Compute minDCF and actDCF
    print(f"Error Rate: {compute_acc_err(PVAL, LVAL)[1]*100:.2f}%, Threshold: {t}")
    min_DCF = compute_minimum_DCF(LLRs, LVAL, eff_prior)
    act_DCF = compute_actual_DCF(eff_prior, compute_confusion_matrix(PVAL, LVAL))
    loss = act_DCF - min_DCF
    percent_loss = loss / min_DCF * 100
    print(f"effPrior={eff_prior}: act_DCF={act_DCF:.3f}, min_DCF={min_DCF:.3f}, loss={percent_loss:.1f}%")
