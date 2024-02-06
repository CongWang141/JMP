import pandas as pd
import numpy as np

def gen_stationary_AR1(n):
    """
    Generate a random n x n matrix A1 for a VAR(1) process
    that satisfies the stationarity condition (all eigenvalues lie inside the unit circle).
    """
    while True:
        # Generate a random matrix
        A1 = np.random.rand(n, n) - 0.5  # Shift to have negative and positive values
        # Check if the generated matrix satisfies the stationarity condition
        eigenvalues = np.linalg.eigvals(A1)
        if np.all(np.abs(eigenvalues) < 1):
            return A1

def data_gen(T0, T1, N_co, N_tr, L, K, drift):  
    """
    Generate data for the simulation.
    T0: int, number of time periods before treatment
    T1: int, number of time periods after treatment
    N_co: int, number of control units
    N_tr: int, number of treated units
    L: int, number of covariates
    K: int, number of factors
    drift: int, drift for the treated units in VAR(1) process
    """

    # Factors for VAR(1) process
    # Assuming a simple structure where each variable depends on its own and the other variable's past value
    A = gen_stationary_AR1(K)
    # Initial values for the first period are drawn from a normal distribution
    F = np.zeros((K, T0+T1))
    F[:, 0] = np.random.normal(size=K)
    # Generate the time series
    for t in range(1, T0+T1):
        F[:, t] = A @ F[:, t-1] + np.random.normal(size=K)

    # Covariates for VAR(1) process
    # Assuming a simple structure where each variable depends on its own and the other variable's past value
    Ai = np.zeros((N_co+N_tr, L, L))
    for i in range(N_co+N_tr):
        Ai[i, :, :] = gen_stationary_AR1(L)
    # Initial values for the first period are drawn from a normal distribution
    X = np.zeros((N_co+N_tr, T0+T1, L))
    X[:, 0, :] = np.random.normal(0, 1, size=(N_co+N_tr, L))

    # Generate the time series for control units
    for i in range(N_co):
        for t in range(1, T0+T1):
             X[i, t, :] = Ai[i, :, :] @ X[i, t-1, :] + np.random.normal(size=L)
    # Generate the time series for treated units with a drift
    for i in range(N_tr):
        for t in range(1, T0+T1):
             X[-i, t, :] = Ai[-i, :, :] @ X[-i, t-1, :] + np.random.normal(drift, 1, size=L)

    # Generate Gama
    Gama = np.random.uniform(0, 0.1, size=(L, K))

    # Generate coefficient Lambda, unit fixed effect alpha, time fixed effect xi
    Lamda = np.random.uniform(0, 1, L)
    alpha = np.random.uniform(0, 1, N_co+N_tr)
    xi = np.random.uniform(0, 1, T0+T1)

    # Generate outcome variable
    Y = np.zeros((N_co+N_tr, T0+T1))
    for t in range(T0+T1):
        for i in range(N_co+N_tr):
            Y[i, t] = X[i, t, :] @ Gama @ F[:, t] # factors and instrumented loadings
            Y[i, t] += X[i, t, :] @ Lamda # effects from covariates
            Y[i, t] += alpha[i] + xi[t] # unit and time fixed effects
            Y[i, t] += np.random.normal(0, 1) # noise

    # Treatment effects and assignment
    tr = np.concatenate([np.zeros(N_co), np.ones(N_tr)])
    delta = np.concatenate([np.zeros(T0), np.arange(1, T1+1) + np.random.normal(0, 1, T1)]) 
    Y += tr.reshape(-1,1) @ delta.reshape(1, -1)
    post_period = np.concatenate([np.zeros(T0), np.ones(T1)])
    # Construct DataFrame
    df = pd.DataFrame({
        'id': np.repeat(np.arange(101, N_co+N_tr + 101), T0+T1),
        'year': np.tile(np.arange(1, T0+T1 + 1), N_co+N_tr),
        'y': Y.flatten(),
        'tr_group': np.repeat(tr, T0+T1),
        'post_period': np.tile(post_period, N_co+N_tr),
        'eff': np.tile(delta, N_co+N_tr)
        })
    for i in range(L):
        df['x' + str(i+1)] = np.tile(X[:, :, i].flatten(), 1)
    df['treated'] = df['tr_group'] * df['post_period']

    return df

# we consider two covariates and two factors
def data_gen_xu(N_co, N_tr, T0, T1, w):
    '''
    N_co: number of control units
    N_tr: number of treated units
    T0: number of pre-treatment periods
    T1: number of post-treatment periods
    w: similarity between treated and control units, 1 means identical
    '''
    # Constants
    N = N_co + N_tr
    T = T0 + T1
    ss = np.sqrt(3)

    # Random components
    epsilon, eta, f, xi = [np.random.normal(0, 1, size=s) for s in [(N, T), (2, N, T), (2, T), T]]
    lambda_co, alpha_co = [np.random.uniform(-ss, ss, size=s) for s in [(2, N_co), N_co]]
    lambda_tr, alpha_tr = [np.random.uniform(ss-2*w*ss, 3*ss-2*w*ss, size=s) for s in [(2, N_tr), N_tr]]

    # Factor loadings and unit fixed effects
    lambda_ = np.concatenate([lambda_co, lambda_tr], axis=1)
    alpha = np.concatenate([alpha_co, alpha_tr])

    # Treatment effects and assignment
    tr = np.concatenate([np.zeros(N_co), np.ones(N_tr)])
    delta = np.concatenate([np.zeros(T0), np.arange(1, T1+1)+ np.random.normal(0, 1, T1)]) 

    # Covariates and outcome
    # here the author didn't mention that he scaled some components in the paper but he did in his code
    x1, x2 = [1 + 0.5*(lambda_.T @ f) + 0.25*(lambda_[0].reshape(-1, 1) + lambda_[1].reshape(-1, 1)) + 0.25*(f[0] + f[1]) + eta[i] for i in range(2)]
    y = tr.reshape(-1,1)@delta.reshape(1, -1) + x1*1 + x2*3 + lambda_.T@f + alpha.reshape(-1,1) + xi + epsilon + 5

    # Construct DataFrame
    df = pd.DataFrame({
        'id': np.repeat(np.arange(101, N + 101), T),
        'year': np.tile(np.arange(1, T + 1), N),
        'y': y.flatten(),
        'tr_group': np.repeat(tr, T),
        'post_period': np.tile(np.concatenate([np.zeros(T0), np.ones(T1)]), N),
        'x1': x1.flatten(),
        'x2': x2.flatten(),
        'eff': np.tile(delta, N),
        })
    df['treated'] = df['tr_group'] * df['post_period']

    return df