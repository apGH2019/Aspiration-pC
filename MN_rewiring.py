import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as scsp
import networkx as nx
import time
import numba as nb
from numba import int64, float64
import seaborn as sns

import PGG
import Record_vectorized as Record
import rewire
import initialize as init

sns.set()

tTime = time.time()
totTime = 0.

totNP = 500
trials = 1
rounds = 1000

eps = 0.02
c = 1.
r = 2.
b = r*c
ini_re = 0.3
h_arr = np.arange(0.001, 1.001, 0.1)
beta_arr = np.geomspace(0.1, 4.0, 10)

h_arr = [0.3]
beta_arr = [3.]
beta_arr = beta_arr[::-1]
beta_arr = np.round(beta_arr, 4)
h_arr = np.round(h_arr, 4)

coopfrac_arr = np.zeros((len(beta_arr), len(h_arr), trials, rounds))
sat_arr = np.zeros((len(beta_arr), len(h_arr), trials, rounds))

hInd = -1
for hab in h_arr:
    hTime = time.time()
    print('hab=', hab)
    hInd = hInd + 1
    betaInd = -1
    for bet in beta_arr:
        betaInd = betaInd + 1
        
        for it in range(trials):

            AdjMat = init.init_adjmat(totNP, ini_re)
            [aspA, satA,
             pcA, payA,
             cpayA, habA,
             betaA, actA] = init.init_arr(totNP, bet, hab)
            
            for i_main in range(rounds):
                print(i_main)
                
                pay = PGG.game(AdjMat, actA, r, c, totNP)

                [coopfrac,
                 sais] = Record.measure(actA, satA, AdjMat)
                coopfrac_arr[betaInd, hInd, it, i_main] = coopfrac
                sat_arr[betaInd, hInd, it, i_main] = sais
                                
                AdjMat = rewire.rewiring_process(AdjMat, actA, 0.3)

                [aspA, satA,
                 pcA, actA,
                 payA, cpayA] = Record.update_iterated(pay, aspA,
                                                       satA, payA,
                                                       cpayA, pcA,
                                                       habA, betaA,
                                                       actA, eps,
                                                       totNP)

        print(totTime/trials)
        print(time.time()-tTime)
        trounds = np.arange(0, rounds, 1)
        plt.plot(trounds, coopfrac_arr[betaInd, hInd, 0, :], 'o-', label=str(bet))
        plt.title('h='+str(hab)+' beta='+str(bet)+' N='+str(totNP))
        plt.xlabel('rounds')
        plt.ylabel('cooperator fraction')
        plt.savefig('coopfrac_N'+str(totNP)+'.png')
        plt.show()
        
        plt.plot(trounds, sat_arr[betaInd, hInd, 0, :], 'o-', label=str(bet))
        plt.title('h='+str(hab)+' beta='+str(bet)+' N='+str(totNP))
        plt.xlabel('rounds')
        plt.ylabel('average satisfaction')
        plt.savefig('avgsat_N'+str(totNP)+'.png')
        plt.show()
        # -------------------------------------------------
    print('h='+str(hab)+' required: ', (time.time()-hTime))
coopfrac_mean = np.mean(np.mean(coopfrac_arr, axis=3), axis=2)
print(time.time() - tTime)
ax = sns.heatmap(coopfrac_mean, vmin=0., vmax=1., cbar_kws={'label': 'Fraction of Coopertors'}, xticklabels=h_arr, yticklabels=beta_arr, cmap='hot')
plt.title('heatmap_betaVShab')
plt.xlabel('h')
plt.ylabel('beta')
#plt.savefig('MN_rewiring_betaVShab_N'+str(totNP)+'_t'+str(rounds)+'.png')
plt.show()
