import numpy as np
import numba as nb

@nb.njit
def init_arr(N, bet, hab, pc = 0.5):
    satisfaction = np.zeros((N, 1))
    aspiration = np.zeros((N, 1))
    pC = np.zeros((N, 1))
    payoff = np.zeros((N, 1))
    cumpayoff = np.zeros((N, 1))
    habituation = np.zeros((N, 1))
    beta = np.zeros((N, 1))
    action = np.zeros((N, 1))
    choose = np.array([0., 1.])
    
    for ind in range(N):
        satisfaction[ind] = 0.
        aspiration[ind] = 9/4.
        beta[ind] = bet
        habituation[ind] = hab
        payoff[ind] = 0.
        pC[ind] = 0.5
        #action[ind] = nb.float64(np.random.random() < 0.5)
        action[ind] = np.random.choice(choose)
        cumpayoff[ind] = 0.
    return(aspiration, satisfaction, pC,
           payoff, cumpayoff, habituation, beta, action)

@nb.njit
def init_adjmat(N, p):
    adjacency_matrix = np.zeros((N, N))
    for i in range(N):
        for j in np.arange(i+1, N, 1):
            response = nb.float64(np.random.random() < p)
            adjacency_matrix[i][j] = response
            adjacency_matrix[j][i] = response
    return(adjacency_matrix)


if __name__ == "__main__":
    print(init_arr(5, 1., 1.))
    print(init_adjmat(5, 0.3))


