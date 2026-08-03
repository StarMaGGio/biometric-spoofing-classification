# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
from src.models.logistic_regression import LogisticRegression, WeightedLogisticRegression
from src.models.utils import split_db_2to1, compute_confusion_matrix
from src.models.bayes_decisions_model import compute_actual_DCF, compute_minimum_DCF
from src.models.utils import quadratic_expansion

def analyze_logistic_regression_with_different_lambdas(D, L):

    # ---------------------
    #  Logistic Regression
    # ---------------------

    # TODO: Add reduced dataset and quadratic expansion analysis

    # Divide the dataset in training and validation sets
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)
    effPrior = 0.1

    inner_menu_option = int(input('\n Choose a model to evaluate:\n'
                                  '1. Logistic Regression\n'
                                  '2. Weighted Logistic Regression\n'
                                  '3. Bayes Error Plot (Different Lambdas/Regularization Parameters)\n'
                                  '0. Back\n'))
    model = ""

    dataset_menu_option = int(input("Choose dataset:\n"
                                  "1. Full dataset\n"
                                  "2. Reduced dataset\n"
                                  "3. Quadratic expanded dataset\n"))
    match dataset_menu_option:
        case 1:
            pass
        case 2:
            percentual_dataset = int(input("Insert the percentual of the dataset to use (from 1 to 99): "))
            step = int(100 / percentual_dataset)
            DTR = DTR[:, ::step]
            LTR = LTR[::step]
        case 3:
            DTR = quadratic_expansion(DTR)
            DVAL = quadratic_expansion(DVAL)
        case _:
            print("Invalid option")
            return  
    
    if inner_menu_option == 0: return

    if inner_menu_option != 3:
        lamb = float(input("Regularization parameter (from 0.0001 to 100.0): "))

    match inner_menu_option:
        case 1:
            # Train model
            LR = LogisticRegression()
            LR.train(DTR, LTR, lamb)
            LLRs = LR.get_log_likelihood_ratios(DVAL)
            # Predict validation labels
            PVAL = LR.predict_binary(DVAL)
        case 2:
            # Train weighted model
            WLR = WeightedLogisticRegression()
            WLR.train(DTR, LTR, lamb)
            LLRs = WLR.get_log_likelihood_ratios(DVAL)
            # Predict validation labels
            PVAL = WLR.predict_binary(DVAL)
        case 3:
            # Plot DCFs for different values of lambda for the selected model
            LLRs = np.zeros(shape=LVAL.shape)
            select_model = int(input("Select model: 1. Logistic Regression, 2. Weighted Logistic Regression"))
            match select_model:
                case 1:
                    model = "Logistic Regression"
                    
                    actDCFs = []
                    minDCFs = []
                    lambs = np.logspace(-4, 2, 13)
                    for lamb in lambs:
                        LR = LogisticRegression()
                        LR.train(DTR, LTR, lamb)
                        LLRs = LR.get_log_likelihood_ratios(DVAL)
                        PVAL = LR.predict_binary(DVAL)

                        minDCF = compute_minimum_DCF(LLRs, LVAL, effPrior, 1.0, 1.0)
                        minDCFs.append(minDCF)
                        print('minDCF: %.4f' % minDCF)

                        conf_matr = compute_confusion_matrix(PVAL, LVAL)
                        actDCF = compute_actual_DCF(effPrior, conf_matr, 1.0, 1.0)
                        actDCFs.append(actDCF)
                        print('actDCF: %.4f' % actDCF)
                    
                        print()
                case 2:
                    model = "Weighted Logistic Regression"

                    actDCFs = []
                    minDCFs = []
                    lambs = np.logspace(-4, 2, 13)
                    for lamb in lambs:
                        WLR = WeightedLogisticRegression()
                        WLR.train(DTR, LTR, lamb, effPrior)
                        LLRs = WLR.get_log_likelihood_ratios(DVAL)
                        PVAL = WLR.predict_binary(DVAL)

                        minDCF = compute_minimum_DCF(LLRs, LVAL, effPrior, 1.0, 1.0)
                        minDCFs.append(minDCF)
                        print('minDCF: %.4f' % minDCF)

                        conf_matr = compute_confusion_matrix(PVAL, LVAL)
                        actDCF = compute_actual_DCF(effPrior, conf_matr, 1.0, 1.0)
                        actDCFs.append(actDCF)
                        print('actDCF: %.4f' % actDCF)
                    
                        print()
                
            plt.figure()
            plt.plot(lambs, minDCFs, label="minDCF", color='r')
            plt.plot(lambs, actDCFs, label="actDCF", color='b')
            plt.xscale('log', base=10)
            plt.ylim([0, 1.7])
            plt.ylabel('DCF value')
            plt.xlabel('Lambda / Regularization')
            plt.title(f"{model} - Different Lambdas/Regularization Parameters")
            plt.show()
            return

    # Compute error rate
    err = (PVAL != LVAL).sum() / float(LVAL.size)
    print('Error rate: %.2f' % (err*100))
    minDCF = compute_minimum_DCF(LLRs, LVAL, effPrior, 1.0, 1.0)
    print('minDCF: %.4f' % minDCF)
