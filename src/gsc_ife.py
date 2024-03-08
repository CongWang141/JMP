import pandas as pd
import numpy as np
import scipy.linalg as sla

# matrix left/right division (following MATLAB function naming)
_mldivide = lambda denom, numer: sla.lstsq(np.array(denom), np.array(numer))[0]
_mrdivide = lambda numer, denom: (sla.lstsq(np.array(denom).T, np.array(numer).T)[0]).T

class gsc_ife(object):
    def __init__(self, df, id, time, outcome, covariates, treated, K):
        """
        df: pd.DataFrame, panel data
        id: str, column name for unit id
        time: str, column name for time period
        outcome: str, column name for outcome variable
        covariates: list of str, column names for covariates
        treated: str, column name for treated unit
        K: int, number of factors

        Initializes the GSC_IFE model with given parameters and data.
        """
        self.df = df
        self.id = id
        self.time = time
        self.outcome = outcome
        self.covariates = covariates
        self.treated = treated
        self.K = K

        # create post_period and tr_group columns
        self.df['post_period'] = self.df.groupby(self.time)[self.treated].transform('max')
        self.df['tr_group'] = self.df.groupby(self.id)[self.treated].transform('max')

        # compute number of treated and control units
        self.N_co = self.df[self.df['tr_group'] == 0][self.id].nunique()
        self.N_tr = self.df[self.df['tr_group'] == 1][self.id].nunique()
        # compute number of pre and post periods
        self.T0 = self.df[self.df['post_period'] == 0]['time'].nunique()
        self.T1 = self.df[self.df['post_period'] == 1]['time'].nunique()
        self.T = self.T0 + self.T1
        self.L = len(covariates)

    def run_gsc_ife(self, verbose=True, MinTol=1e-6, MaxIter=1e3):
        """
        Main method to run the GSC_IFE estimation.
        """
        # step 1: estimate F and beta use control units
        # compute X0, Y0 to estimate F and beta
        Y0, X0 = self._prepare_matrices(self.df.query("tr_group==0"))

        # Initialize parameters
        N, T, L = X0.shape
        F0 = np.random.randn(T, self.K)
        lambda0 = np.random.randn(N, self.K)
        beta0 = np.zeros(L)

        # previous interation objective function value
        prev_obj = float('inf')
        tol, iter = float('inf'), 0
        while iter < MaxIter and tol > MinTol:
            beta1, lambda1, F1 = self.ife_est(Y0, X0, F0, lambda0, self.K)

            # compute the objective function
            obj_fun = np.linalg.norm(Y0 - X0@beta1 - lambda1@F1.T)**2
            tol = abs(obj_fun - prev_obj)

            # update parameters
            beta0, lambda0, F0 = beta1, lambda1, F1
            prev_obj = obj_fun

            if verbose:
                print(f'iter: {iter}, tol: {tol}')
            iter += 1

        # step 2: use treated units before treatment to compute lambda
        Y10, X10 = self._prepare_matrices(self.df.query("tr_group==1 & post_period==0"))
        N_co, T0 = Y10.shape
        numer = F1[:T0, :].T@(Y10 - (X10@beta1)).T
        denom = F1[:T0, :].T@F1[:T0, :]
        lambda1 = _mldivide(denom, numer).T

        # step 3: compute Y_syn
        # get X1 to compute Y_syn
        Y1, X1 = self._prepare_matrices(self.df.query("tr_group==1"))
        self.Y_syn = X1@beta1 + lambda1@F0.T
    
    def ife_est(self, Y0, X0, F0, lambda0, K):
        """
        ALS method to estimate the IFE model.
        """
        N, T, L = X0.shape

        # flatten X and Y
        y = Y0.flatten()
        x = X0.reshape(N*T, L)

        # compute beta1
        denom = x.T @ x
        numer = x.T @ (y - (lambda0@F0.T).flatten())
        beta1 = _mldivide(denom, numer)

        # compute lambda1 and F1
        residual = (y - x@beta1).reshape(N, T)
        M = (residual.T @ residual) / (N*T)
        s, v, d = sla.svd(M) # sigular value decomposition
        F1 = s[:, :K]
        lambda1 = residual @ F1 / T

        return beta1, lambda1, F1
    
    def _prepare_matrices(self, df):
        """
        define a function to pivot data
        """
        Y = df.pivot(index=self.id, columns=self.time, values=self.outcome).values
        X = np.array([df.pivot(index=self.id, columns=self.time, values=x).values for x in self.covariates]).transpose(1, 2, 0)

        return Y, X
    
    def fit(self, verbose=True, MinTol=1e-6, MaxIter=1e3):
        """
        Fit the model to the data.
        """
        self.run_gsc_ife(verbose=verbose, MinTol=MinTol, MaxIter=MaxIter)
        return self


        

        