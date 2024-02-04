import pandas as pd
import numpy as np
import scipy.linalg as sla
import scipy.sparse.linalg as ssla

# matrix left/right division (following MATLAB function naming)
_mldivide = lambda denom, numer: sla.lstsq(np.array(denom), np.array(numer))[0]
_mrdivide = lambda numer, denom: (sla.lstsq(np.array(denom).T, np.array(numer).T)[0]).T

class GSC_IPCA(object):
    def __init__(self, df, id, year, outcome, covariates, tr_group, post_period, K, L):
        """
        df: pd.DataFrame, panel data
        id: str, column name for unit id
        year: str, column name for time period
        outcome: str, column name for outcome variable
        covariates: list of str, column names for covariates
        tr_group: str, column name for treatment group
        post_period: str, column name for post period
        K: int, number of factors
        L: int, number of covariates
        """
        self.df = df
        self.id = id
        self.year = year
        self.outcome = outcome
        self.covariates = covariates
        self.tr_group = tr_group
        self.post_period = post_period
        self.K = K
        self.L = L

    def run_gsc_ipca(self):
        Y0 = self.df[self.df[self.tr_group] == 0].pivot(index=self.id, columns=self.year, values=self.outcome)
        X0 = np.zeros((self.N_co, self.T0+self.T1, self.L))
        for i in range(self.N_co):
            for t in range(self.T0):
                X0[i, t, :] = self.df[self.df[self.id] == i][self.covariates].iloc[t]
        # step 1: initial guess
        svU, svS, svV = ssla.svds(Y0, self.K)
        svU, svS, svV = np.fliplr(svU), svS[::-1], np.flipud(svV) # reverse order to match MATLAB svds output
        # initial guess for factors
        F0 = np.diag(svS) @ svV
        gamma0 = np.random.normal(0, 1, size=(self.L, self.K))

    def als_est(self, F0):
        # step L: compute Gamma, 
        vec_len = self.L*self.K
        numer, denom = np.zeros(vec_len), np.zeros((vec_len, vec_len))
        for t in range(self.T0+self.T1):
            ft_slice = F0[:, t]  # Kx1 for each t
            for i in range(self.N_co):
                X0_slice = X0[i, t, :]  # Lx1 for each i, t
                # Compute Kronecker product
                kron_prod = np.kron(X0_slice, ft_slice)
                # Update denom and numer
                denom += np.outer(kron_prod, kron_prod)
                numer += kron_prod * Y0[i, t]

        # Solve for Gamma using the computed matrices
        gamma_hat = _mldivide(denom, numer)

        # Reshape gamma_hat if necessary to match the expected dimensions of Gamma
        gamma1 = gamma_hat.reshape(L, K)

        # step 3: update F
        F1 = np.zeros((K, T0+T1))
        for t in range(T0+T1):
            denom = gamma1.T@X0[:,t,:].T@X0[:,t,:]@gamma1
            numer = gamma1.T@X0[:,t,:].T@Y0[:,t]
            F1[:, t] = _mldivide(denom, numer)
        return F1, gamma1
