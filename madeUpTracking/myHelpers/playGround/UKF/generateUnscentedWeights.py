import sys
sys.path.append("../")
import commonVariables as commonVar

print_generateUnscentedWeights = True

def print_(*element):
    if(print_generateUnscentedWeights):
        print(element)


def generateUnscentedWeights(L, alpha, beta, kappa): #checkCount : 1

    """
        Description:
            [WR00 Equation 15]
        Input:
            L: float
            alpha: float
            beta: float
            kappa: float
    """

    lambda_ = (alpha ** 2) * (L + kappa) - L


    Ws0 = lambda_ / (L + lambda_)
    Wc0 = lambda_ / (L + lambda_) + (1 - alpha**2 + beta)

    Wsi = 0.5 / (L + lambda_)
    Wci = 0.5 / (L + lambda_)

    return ([Ws0, Wsi],[Wc0, Wci], lambda_)




Ws, Wc, lambda_ = generateUnscentedWeights(commonVar.L, commonVar.alpha, commonVar.beta, commonVar.kappa)
print_("Ws : ",Ws)
print_("Wc : ", Wc)
print_("lambda : ", lambda_)