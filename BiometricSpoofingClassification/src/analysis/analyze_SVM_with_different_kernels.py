# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
from src.models.support_vector_machines import SupportVectorMachine, KernelSupportVectorMachine
from src.models.utils import split_db_2to1, compute_confusion_matrix, polyKernel, rbfKernel
from src.models.bayes_decisions_model import compute_actual_DCF, compute_minimum_DCF

def analyze_SVM_with_different_kernels(D, L):

    # -------------------------
    #  Support Vector Machines
    # -------------------------

    # Divide the dataset in training and validation sets
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)

    inner_menu_option = int(input('\n Choose a model to evaluate:\n'
                                  '1. Linear Support Vector Machine\n'
                                  '2. Linear Support Vector Machine (Centered Data)\n'
                                  '3. Support Vector Machine Polynomial Kernel\n'
                                  '4. Support Vector Machine RBF Kernel\n'
                                  '0. Back\n'))
    model = ""

    if inner_menu_option == 0: return

    dataset_menu_option = int(input("Choose dataset:\n"
                                  "1. Full dataset\n"
                                  "2. Reduced dataset\n"))
    match dataset_menu_option:
        case 1:
            pass
        case 2:
            percentual_dataset = int(input("Insert the percentual of the dataset to use (from 1 to 99): "))
            step = int(100 / percentual_dataset)
            DTR = DTR[:, ::step]
            LTR = LTR[::step]
        case _:
            print("Invalid option")
            return  

    minDCFs = []
    actDCFs = []

    match inner_menu_option:
        case 1:
            model = "SVM Linear"
            Cs = np.logspace(-5, 0, 11)   
            minDCFs.clear()
            actDCFs.clear()
            for C in Cs:
                SVM = SupportVectorMachine()
                SVM.train(DTR, LTR, C, K=1.0)
                minDCFs.append(compute_minimum_DCF(SVM.get_scores(DVAL), LVAL, 0.1, 1.0, 1.0))
                actDCFs.append(compute_actual_DCF(0.1, compute_confusion_matrix(SVM.predict(DVAL), LVAL), 1.0, 1.0))
        case 2:
            model = "SVM Linear Centered Data"
            Cs = np.logspace(-5, 0, 11)   
            minDCFs.clear()
            actDCFs.clear()
            # Center Dataset
            mu = DTR.mean(1).reshape((DTR.shape[0], 1))
            DTR_centered = DTR - mu
            DVAL_centered = DVAL - mu
            for C in Cs:
                SVM = SupportVectorMachine()
                SVM.train(DTR_centered, LTR, C, K=1.0)
                minDCFs.append(compute_minimum_DCF(SVM.get_scores(DVAL_centered), LVAL, 0.1, 1.0, 1.0))
                actDCFs.append(compute_actual_DCF(0.1, compute_confusion_matrix(SVM.predict(DVAL_centered), LVAL), 1.0, 1.0))
        case 3:
            degree = int(input('Degree of the polynomial kernel (ex. 2): '))
            model = f"SVM Polynomial Kernel (degree = {degree})"
            kernelFunc = polyKernel(degree, 1)
            eps = 0.0
            Cs = np.logspace(-5, 0, 11)
            minDCFs.clear()
            actDCFs.clear()
            for C in Cs:
                KSVM = KernelSupportVectorMachine()
                KSVM.train(DTR, LTR, C, kernelFunc=kernelFunc, eps=eps)
                minDCFs.append(compute_minimum_DCF(KSVM.get_scores(DVAL), LVAL, 0.1, 1.0, 1.0))
                actDCFs.append(compute_actual_DCF(0.1, compute_confusion_matrix(KSVM.predict(DVAL), LVAL), 1.0, 1.0))
        case 4:
            gamma_exp = float(input('Gamma exponent for the RBF kernel (ex. -4 for gamma = e^-4): '))
            gamma = np.exp(gamma_exp)
            model = f"SVM RBF Kernel (gamma = e^{gamma_exp})"
            minDCFs.clear()
            actDCFs.clear()
            eps = 1.0
            Cs = np.logspace(-3, 2, 11)
            kernelFunc = rbfKernel(gamma)
            for C in Cs:
                KSVM = KernelSupportVectorMachine()
                KSVM.train(DTR, LTR, C, kernelFunc=kernelFunc, eps=eps)
                minDCFs.append(compute_minimum_DCF(KSVM.get_scores(DVAL), LVAL, 0.1, 1.0, 1.0))
                actDCFs.append(compute_actual_DCF(0.1, compute_confusion_matrix(KSVM.predict(DVAL), LVAL), 1.0, 1.0))

    # Plot actDCF and minDCF for different values of C for the selected model
    plt.figure()
    plt.plot(Cs, minDCFs, label="minDCF", color='r')
    plt.plot(Cs, actDCFs, label="actDCF", color='b')
    plt.xscale('log', base=10)
    plt.ylim([0, 1.7])
    plt.ylabel('DCF value')
    plt.xlabel('C value')
    plt.title(f"{model}")
    plt.legend()
    plt.show()
    print()
