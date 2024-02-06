import pandas as pd
import numpy as np
import scipy.linalg as sla

class gsc_ife(object):
    def __init__(self, df, id, year, outcome, covariates, treated, K):
        """
        df: pd.DataFrame, panel data
        id: str, column name for unit id
        year: str, column name for time period
        outcome: str, column name for outcome variable
        covariates: list of str, column names for covariates
        treated: str, column name for treated unit
        K: int, number of factors

        Initializes the GSC_IFE model with given parameters and data.
        """
        self.df = df
        self.id = id
        self.year = year
        self.outcome = outcome
        self.covariates = covariates
        self.treated = treated
        self.K = K

        # create post_period and tr_group columns
        self.df['post_period'] = self.df.groupby(self.year)[self.treated].transform('max')
        self.df['tr_group'] = self.df.groupby(self.id)[self.treated].transform('max')
        self.df['const'] = 1
        self.covariates_with_const = self.covariates + ['const']

        # compute number of treated and control units
        # compute number of treated and control units
        self.N_co = self.df[self.df['tr_group'] == 0][self.id].nunique()
        self.N_tr = self.df[self.df['tr_group'] == 1][self.id].nunique()
        # compute number of pre and post periods
        self.T0 = self.df[self.df['post_period'] == 0]['year'].nunique()
        self.T1 = self.df[self.df['post_period'] == 1]['year'].nunique()
        self.T = self.T0 + self.T1
        self.L = len(covariates)

        # compute Y10 and X10 to estimate L1: factor loading for treated units
        # prepare pre-treatment treated data for estimation -- L1(factor loading for treated units)
        self.Y10_wide = self.df[(self.df['tr_group']==1) & (self.df['post_period']==0)].pivot(index=self.year, columns=self.id, values=self.outcome).values
        self.X10_wide = np.empty((self.L+1, self.N_tr, self.T0))

        iter = 0
        for x in self.covariates_with_const:
            self.X10_wide[iter, :, :] = self.df[(self.df['tr_group']==1) & (self.df['post_period']==0)].pivot(index=self.id, columns=self.year, values=x).values
            iter += 1

        # compute Y0 and X0 wide format
        # prepare contorl data for estimation--beta, F
        Y0_wide = self.df.query("tr_group==0").pivot(index=self.year, columns=self.id, values=self.outcome).values
        X0_wide = np.empty((self.L+1, self.N_co, self.T))

        iter = 0
        for x in self.covariates_with_const:
            X0_wide[iter, :, :] = self.df.query("tr_group==0").pivot(index=self.id, columns=self.year, values=x).values
            iter += 1

        self.Y0_wide, self.X0_wide = Y0_wide, X0_wide

    def run_gsc_ife(self, verbose=True, MinTol=1e-6, MaxIter=1e3):
        """
        Main method to run the GSC_IFE estimation.
        """

        # compute Y0 and X0, to estimate Fac and Gamma_ctrl
        Y0 = self.df[self.df['tr_group'] == 0][self.outcome].values
        X0 = self.df[self.df['tr_group'] == 0][self.covariates_with_const].values

        prev_fun_val = float('inf')
        # estimate the IFE model
        iter, tol = 0, float('inf')
        while iter < MaxIter and tol > MinTol:
            beta, F1, L1 = self.ife_est(Y0, X0)
            # compute the objective function
            obj_fun = np.sum((self.Y0_wide-(self.X0_wide.T@beta).reshape(self.T, self.N_co) - F1@L1.T).T @ (self.Y0_wide-(self.X0_wide.T@beta).reshape(self.T, self.N_co) - F1@L1.T)**2)
            tol = abs(prev_fun_val - obj_fun)
            prev_fun_val = obj_fun
            if verbose:
                print(f'iter {iter}: tolFac = {tol:.2e}')
            iter += 1
        # store the estimated parameters    
        self.beta, self.F1 = beta, F1

        # with treated units pre-period to compute factor loading for treated units
        F0 = self.F1[:self.T0, ]
        L1 = np.linalg.inv(F0.T @ F0) @ F0.T @ (self.Y10_wide - (self.X10_wide.T @ self.beta).reshape(self.T0, self.N_tr))
        self.L1 = L1
    
    def ife_est(self, Y0, X0):
        """
        Method to estimate the IFE model.
        """
        # initialize the factor and factor loadings
        F0 = np.random.normal(0, 1, size=(self.T, self.K))
        L0 = np.random.normal(0, 1, size=(self.N_co, self.K))

        # compute beta
        beta = np.linalg.inv(X0.T @ X0) @ (X0.T @ (Y0 - (F0@L0.T).flatten()))

        # update F0 and L0
        residual = (self.Y0_wide - (self.X0_wide.T @ beta).reshape(self.T, self.N_co))
        M = (residual @ residual.T) / (self.N_co*self.T)
        svU, svS, svV = sla.svd(M)
        F1 = svU[:, :self.K]
        L1 = residual.T @ F1 / self.T

        return beta, F1, L1
    
    def fit(self, verbose=True, MinTol=1e-6, MaxIter=1e3):
        """
        Method to fit the GSC_IFE model.
        """
        self.run_gsc_ife(verbose=verbose, MinTol=MinTol, MaxIter=MaxIter)

    # reshape covariates X for treated units all time periods
        X1 = np.zeros((len(self.covariates_with_const), self.N_tr, self.T))
        iter = 0
        for x in self.covariates_with_const:
            X1[iter, :, :] = self.df[self.df['tr_group'] == 1].pivot(index=self.id, columns=self.year, values=x).values
            iter += 1
        Y_syn = (X1.T @ self.beta).reshape(self.T, self.N_tr) + self.F1 @ self.L1
        self.Y_syn = Y_syn
        return self


        

        