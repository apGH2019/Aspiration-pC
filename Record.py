import numpy as np
import numba as nb
from numba import float64
from numba import guvectorize


@nb.njit
def measure(action, sat, AM):
    satisfaction = 0.
    for s in sat:
        satisfaction = satisfaction + s[0]
    return(np.sum(action)/len(sat), satisfaction/len(sat))

'''
@nb.guvectorize([(float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:])], '(n),(n)->(n)')
def update(payoff, asp, sat, pay, cpay, pC, hab, beta, act, tremble):
    pay = payoff
    cpay = cpay + payoff

    sat = np.tanh(beta * (pay - asp))

    if sat >= 0. and act >= 0.5:
        pC = pC + (1 - pC) * sat
    elif sat < 0. and act >= 0.5:
        pC = pC + sat * pC
    elif sat >= 0. and act < 0.5:
        pC = pC - sat * pC
    else:
        pC = pC - (1 - pC) * sat

    if np.random.random() > tremble:
        act = np.random.random() <= pC
    else:
        act = np.random.random() > pC

    asp = (1 - hab) * asp + hab * pay


    #return(asp, sat, pC, act)
'''

@nb.njit(debug=True)
def update_iterated(payoff, asp, sat, pay, cpay, pC, hab, beta, act, tremble, N):

    aspM = np.zeros_like(asp)
    satM = np.zeros_like(sat)
    pCM = np.zeros_like(pC)
    actM = np.zeros_like(act)
    payM = np.zeros_like(pay)
    cpayM = np.zeros_like(cpay)
    
    for index in range(N):
        payM[index] = payoff[index]
        cpayM[index] = cpay[index] + payoff[index]

        satM[index] = np.tanh(beta[index] *
                           (payM[index] - asp[index]))

        if satM[index] >= 0. and act[index] >= 0.5:
            pCM[index] = pC[index] + (1 - pC[index]) * satM[index]
        elif satM[index] < 0. and act[index] >= 0.5:
            pCM[index] = pC[index] + satM[index] * pC[index]
        elif satM[index] >= 0. and act[index] < 0.5:
            pCM[index] = pC[index] - satM[index] * pC[index]
        else:
            pCM[index] = pC[index] - (1. - pC[index]) * satM[index]

        if np.random.random() > tremble:
            actM[index] = np.random.random() <= pCM[index]
        else:
            actM[index] = np.random.random() > pCM[index]

        aspM[index] = (1 - hab[index]) * asp[index] + hab[index] * payM[index]


    return(aspM, satM, pCM, actM, payM, cpayM)


if __name__ == "__main__":
    act = np.ones(5)
    sats = np.ones(5)
    act[3]=0.
    sats[2]=0.
    am = 0.
    mea = measure(act, sats, am)
    print(mea)
    a = np.array([1,2,3,4])
    b = np.array([100,200,300,400])
    print(update(a, b))
