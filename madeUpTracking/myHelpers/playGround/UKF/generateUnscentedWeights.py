print_generateUnscentedWeights = False

def print_(*element):
    if(print_generateUnscentedWeights):
        print(element)


def generateUnscentedWeights(L, alpha, beta, kappa):

    lambda_ = (alpha ** 2) * (L + kappa) - L


    Ws0 = lambda_ / (L + lambda_)
    Wc0 = lambda_ / (L + lambda_) + (1 - alpha**2 + beta)

    Wsi = 0.5 / (L + lambda_)
    Wci = 0.5 / (L + lambda_)

    return ([Ws0, Wsi],[Wc0, Wci], lambda_)


L = 5
alpha = 0.01
beta = 2
kappa = 0

Ws, Wc, lambda_ = generateUnscentedWeights(L, alpha, beta, kappa)
print_("Ws : ",Ws)
print_("Wc : ", Wc)
print_("lambda : ", lambda_)