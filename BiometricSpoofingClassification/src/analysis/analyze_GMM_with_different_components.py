# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
# pyrefly: ignore [missing-import]
import numpy as np
from src.models.gaussian_mixture_models import GaussianMixtureModel
from src.models.utils import split_db_2to1, compute_confusion_matrix
from src.models.bayes_decisions_model import compute_minimum_DCF

def analyze_GMM_with_different_components(D, L):

    # -------------------------
    #  Gaussian Mixture Models
    # -------------------------


    # Divide the dataset in training and validation sets
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)

    inner_menu_option = int(input('\n Choose a model to evaluate:\n'
                                  '1. Gaussian Mixture Model\n'
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

    match inner_menu_option:
        case 1:
            # num_components = int(input("Enter the number of components for the GMM (1, 2, 4, 8, 16): "))
            # alpha = float(input("Enter the alpha parameter for the GMM (ex 0.1): "))
            # psi = float(input("Enter the psi parameter for the GMM (ex. 0.01): "))

            alpha=0.1
            psi=0.01
            minDCFs = []
            for num_components in [1, 2, 4, 8, 16]:
                print(f"\nTraining now with {num_components} components...")

                eff_prior = 0.1
                t = np.log((1-eff_prior)/eff_prior)

                GMM = GaussianMixtureModel()
                GMM.train(DTR, LTR, numComponents=num_components, alpha=alpha, psi=psi)
                # PVAL = GMM.predict(DVAL, t)

                minDCF = compute_minimum_DCF(GMM.get_scores(DVAL), LVAL, eff_prior, 1.0, 1.0)
                minDCFs.append(minDCF)

                print(f"\nComponents: {num_components}: minimum DCF: {minDCF:.4f}")

            plt.figure()
            plt.plot([1, 2, 4, 8, 16], minDCFs)
            plt.title("minDCF for different number of components in GMM")
            plt.xlabel("Number of components")
            plt.ylabel("minDCF")
            plt.xticks([1, 2, 4, 8, 16])
            plt.ylim([0.0, 0.4])
            plt.show()
            
