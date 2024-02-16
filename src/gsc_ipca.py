import pandas as pd
import numpy as np
import scipy.linalg as sla
import scipy.sparse.linalg as ssla

# matrix left/right division (following MATLAB function naming)
_mldivide = lambda denom, numer: sla.lstsq(np.array(denom), np.array(numer))[0]
_mrdivide = lambda numer, denom: (sla.lstsq(np.array(denom).T, np.array(numer).T)[0]).T

class gsc_ipca(object):
    def __init__(self, df, id, time, outcome, covariates, treated, K, L):
        """
        df: pd.DataFrame, panel data
        id: str, column name for unit id
        time: str, column name for time period
        outcome: str, column name for outcome variable
        covariates: list of str, column names for covariates
        treated: str, column name for treated unit
        K: int, number of factors
        L: int, number of covariates
        """
        # Initializes the GSC_IPCA model with given parameters and data.
        self.df = df
        self.id = id
        self.time = time
        self.outcome = outcome
        self.covariates = covariates
        self.treated = treated
        self.K = K
        self.L = L

        # create post_period and tr_group columns
        self.df['post_period'] = self.df.groupby(self.time)[self.treated].transform('max')
        self.df['tr_group'] = self.df.groupby(self.id)[self.treated].transform('max')

        # compute number of treated and control units
        self.N_co = self.df[self.df['tr_group'] == 0][self.id].nunique()
        self.N_tr = self.df[self.df['tr_group'] == 1][self.id].nunique()
        # compute number of pre and post periods
        self.T0 = self.df[self.df['post_period'] == 0][self.time].nunique()
        self.T1 = self.df[self.df['post_period'] == 1][self.time].nunique()

    def run_gsc_ipca(self, verbose=True, MinTol=1e-6, MaxIter=1e3):
        """
        Main method to run the GSC_IPCA estimation.
        """
    
        # compute Y0 and X0, to estimate Fac and Gamma_ctrl
        Y0, X0 = self._prepare_matrices(self.df[self.df['tr_group'] == 0])
        # compute Y10 and X10, to estimate Gamma_treat
        Y10, X10 = self._prepare_matrices(self.df[(self.df['tr_group'] == 1) & (self.df['post_period'] == 0)])

        # step 1: compute Fac and Gamma_ctrl
        # initial guess for F0 and Gamma0
        svU, svS, svV = ssla.svds(Y0, self.K)
        svU, svS, svV = np.fliplr(svU), svS[::-1], np.flipud(svV) # reverse order to match MATLAB svds output
        # initial guess for factors
        F0 = np.diag(svS) @ svV

        Gamma0 = np.random.normal(0, 1, size=(self.L, self.K))
        
        iter, tol = 0, float('inf')
        # ALS estimate
        while iter < MaxIter and tol > MinTol:
            Fac, Gamma_ctrl = self.als_est(F0, Y0, X0)
            tol_Gamma = abs(Gamma_ctrl - Gamma0).max()
            tol_F = abs(Fac - F0).max()
            tol = max(tol_Gamma, tol_F)

            if verbose:
                print('iter {}: tolGamma = {} and tolFac = {}'.format(iter, tol_Gamma, tol_F))
            F0, Gamma0 = Fac, Gamma_ctrl
            iter += 1
        # store the result
        self.Gamma_ctrl, self.Fac = Gamma_ctrl, Fac

        # step 2: compute Gamma_treat
        vec_len = self.L*self.K
        numer, denom = np.zeros(vec_len), np.zeros((vec_len, vec_len))
        for t in range(self.T0):
            ft_slice = self.Fac[:, t]  # Kx1 for each t
            for i in range(self.N_tr):
                X10_slice = X10[i, t, :]  # Lx1 for each i, t
                # Compute Kronecker product
                kron_prod = np.kron(X10_slice, ft_slice)
                # Update denom and numer
                denom += np.outer(kron_prod, kron_prod)
                numer += kron_prod * Y10[i, t]
        # Solve for Gamma using the computed matrices
        Gamma_treat = _mldivide(denom, numer)
        Gamma_treat = Gamma_treat.reshape(self.L, self.K)

        # step 3: compute Y_syn
        # get X1 to compute Y_syn
        Y1, X1 = self._prepare_matrices(self.df[self.df['tr_group'] == 1])
        
        # compute Y_syn
        Y_syn = np.zeros((self.N_tr, self.T0+self.T1))
        for t in range(self.T0+self.T1):
            for n in range(self.N_tr):
                Y_syn[n, t] = X1[n, t, :]@Gamma_treat@self.Fac[:, t]
        self.Y_syn = Y_syn

    
    def _prepare_matrices(self, df):
        Y = df.pivot(index=self.id, columns=self.time, values=self.outcome).values
        X = np.array([df.pivot(index=self.id, columns=self.time, values=x).values for x in self.covariates]).transpose(1, 2, 0)

        return Y, X

    def als_est(self, F0, Y0, X0):
        """
        Alternate Least Squares estimation for Gamma and F matrices.
        """
        # step 2: with F0 compute Gamma, 
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
        Gamma_hat = _mldivide(denom, numer)

        # Reshape Gamma_hat if necessary to match the expected dimensions of Gamma
        Gamma1 = Gamma_hat.reshape(self.L, self.K)

        # step 3: update F
        F1 = np.zeros((self.K, self.T0+self.T1))
        for t in range(self.T0+self.T1):
            denom = Gamma1.T@X0[:,t,:].T@X0[:,t,:]@Gamma1
            numer = Gamma1.T@X0[:,t,:].T@Y0[:,t]
            F1[:, t] = _mldivide(denom, numer)
        return F1, Gamma1
    
    def fit(self, verbose=True, MinTol=1e-6, MaxIter=1e3):
        """
        Fit the model to the data.
        """
        self.run_gsc_ipca(verbose=verbose, MinTol=MinTol, MaxIter=MaxIter)
        return self