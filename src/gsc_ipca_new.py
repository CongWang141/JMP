import numpy as np
import scipy.linalg as sla
import scipy.sparse.linalg as ssla

# matrix left/right division (following MATLAB function naming)
_mldivide = lambda denom, numer: sla.lstsq(np.array(denom), np.array(numer))[0]
_mrdivide = lambda numer, denom: (sla.lstsq(np.array(denom).T, np.array(numer).T)[0]).T

class gsc_ipca_new(object):
    def __init__(self, df, id, time, outcome, covariates, treated, K):
        """
        df: pd.dataframe, panel data
        id: str, column name for unit id
        time: str, column name for time period
        outcome: str, column name for outcome variable
        covariates: list of str, column names for covariates
        treated: str, column name for treated unit
        K: int, number of factors
        """
        # initializes the GSC_IPCA_NEW model with given parameters and data.
        self.df = df
        self.id = id
        self.time = time
        self.outcome = outcome
        self.covariates = covariates
        self.treated = treated
        self.K = K

        # gen number of covariates 
        self.L = len(covariates)

        # gen post_period and tr_group variables
        self.df['post_period'] = self.df.groupby(self.time)[self.treated].transform('max')
        self.df['tr_group'] = self.df.groupby(self.id)[self.treated].transform('max')

        # gen number of treated and control units
        self.N_co = self.df.query("tr_group==0")[self.id].nunique()
        self.N_tr = self.df.query("tr_group==1")[self.id].nunique()

        # gen number of pre and post periods
        self.T0 = self.df.query("post_period==0")[self.time].nunique()
        self.T1 = self.df.query("post_period==1")[self.time].nunique()

    def run_gsc_ipca(self, verbose=True, MinTol=1e-6, MaxIter=1e3):
        """
        Main method to run the GSC_IPCA estimation
        """
        # gen Y0 and X0, to estimate Factors and Gamma
        Y0, X0 = self._prepare_matrices(self.df.query("tr_group==0"))

        # step 1: compute F, gamma, and lambda
        svU, svS, svV = ssla.svds(Y0, self.K)
        svU, svS, svV = np.fliplr(svU), svS[::-1], np.flipud(svV) # reverse order to match MatLab svds output
        # initial guess for F0
        F0 = np.diag(svS) @ svV

        # initialized gamma0 and lambda0
        gamma0 = np.random.normal(0, 1, size=(self.L, self.K))
        lambda0 = np.random.normal(0, 1, size=(self.N_co, self.K))

        iter, tol = 0, float('inf')
        # ALS estimate
        while iter < MaxIter and tol > MinTol:
            F1, gamma1, lambda1 = self.als_est(F0, lambda0, Y0, X0)
            tol_F = abs(F1 - F0).max()
            tol_gamma = abs(gamma1 - gamma0).max()
            tol_lambda = abs(lambda1 - lambda0).max()
            tol = max(tol_F, tol_gamma, tol_lambda)

            if verbose:
                 print('iter {}: tol_F = {}, tol_gamma = {}, tol_lambda = {}'.format(iter, tol_F, tol_gamma, tol_lambda))
            F0, gamma0, lambda0 = F1, gamma1, lambda1
            iter += 1
        # store the reault
        self.F1, self.gamma1, self.lambda1 = F1, gamma1, lambda1

        # step 2: compute lambda with treated group before treatment period
        Y10, X10 = self._prepare_matrices(self.df.query("tr_group==1 & post_period==0"))

        residual = np.zeros((self.N_tr, self.T0))
        for t in range(self.T0):
            residual[:, t] = Y10[:, t] - X10[:,t,:]@self.gamma1@self.F1[:, t]
        M = residual.T@residual / (self.N_tr*self.T0)
        svU, svS, svV = sla.svd(M) 
        F_ = svU[:, :self.K]
        lambda10_1 = residual @ F_ / self.T0

        # step 3: compute the counterfactuals
        Y1, X1 = self._prepare_matrices(self.df.query("tr_group==1"))

        T = self.T0 + self.T1
        Y_syn = np.zeros((self.N_tr, T))
        for t in range(T):
            for i in range(self.N_tr):
                Y_syn[i, t] = X1[i,t,:]@self.gamma1@self.F1[:, t] + lambda10_1[i, :]@self.F1[:, t]

        # store the result
        self.Y_syn = Y_syn

    def _prepare_matrices(self, df):
        """
        define a function to generate pivoted data
        """
        Y = df.pivot(index=self.id, columns=self.time, values=self.outcome).values
        X = np.array([df.pivot(index=self.id, columns=self.time, values=x).values for x in self.covariates]).transpose(1, 2, 0)
        return Y, X
    
    def als_est(self, F0, lambda0, Y0, X0):
        """
        Alternate Least Squares estimation for Gamma and F
        """
        # gen total time period
        T = self.T0 + self.T1

        # with lambda and F, update gamma
        vec_len = self.L*self.K
        numer, denom = np.zeros(vec_len), np.zeros((vec_len, vec_len))
        for t in range(T):
            ft_slice = F0[:, t] # Kx1 for each t
            for i in range(self.N_co):
                lambda_slice = lambda0[i, :] # 1xK for each i
                x0_slice = X0[i,t,:] # 1xL for each i
                # compute the kronecker product
                kron_prod = np.kron(x0_slice, ft_slice)
                # update the numerator and denominator
                numer += kron_prod * (Y0[i,t] - lambda_slice @ ft_slice)
                denom += np.outer(kron_prod, kron_prod)
        # solve for gamma using the numerator and denominator
        gamma1 = _mldivide(denom, numer).reshape(self.L, self.K)

        # with F and gamma, update lambda
        residual = np.zeros((self.N_co, T))
        for t in range(T):
            residual[:,t] = Y0[:,t] - X0[:,t,:]@gamma1@F0[:,t]
        M = residual.T@residual / (self.N_co*T)
        s, v, d = sla.svd(M)
        F_ = s[:, :self.K]
        lambda1 = residual @ F_ / T

        # with gamma and lambda, update F
        F1 = np.zeros((self.K, T))
        for t in range(T):
            sudo_x = X0[:,t,:]@gamma1 + lambda1
            denom = sudo_x.T@sudo_x
            numer = sudo_x.T@Y0[:,t]
            F1[:,t] = _mldivide(denom, numer)
            
        return F1, gamma1, lambda1
        
    def fit(self, verbose=True, MinTol=1e-6, MaxIter=1e3):
        """
        Fit the model to the data.
        """
        self.run_gsc_ipca(verbose=verbose, MinTol=MinTol, MaxIter=MaxIter)
        return self



        
