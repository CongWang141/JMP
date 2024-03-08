import numpy as np
import scipy.linalg as sla
import scipy.sparse.linalg as ssla

# matrix left/right division (following MATLAB function naming)
_mldivide = lambda denom, numer: sla.lstsq(np.array(denom), np.array(numer))[0]
_mrdivide = lambda numer, denom: (sla.lstsq(np.array(denom).T, np.array(numer).T)[0]).T

class gsc_ipca_2(object):
    def __init__(self, df, id, time, outcome, covariates, treated, K):
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

        # create post_period and tr_group columns
        self.df['post_period'] = self.df.groupby(self.time)[self.treated].transform('max')
        self.df['tr_group'] = self.df.groupby(self.id)[self.treated].transform('max')

    def run_gsc_ipca2(self, MaxIter=1e3, MinTol=1e-6, verbose=True):
        """
        Main method to run the GSC-IPCA2 estimation.
        """     
        ## step 1: estimate F for the whole time period
        # compute Y0, X0 to estimate F and Gama 
        Y0, X0 = self._prepare_matrices(self.df.query("tr_group==0"))
        # dataset demision 
        N_co, T, L = X0.shape

        # step a: initial guess for F0 and Gama0
        svU, svS, svV = ssla.svds(Y0, self.K)
        svU, svS, svV = np.fliplr(svU), svS[::-1], np.flipud(svV) # reverse the order from largest to smallest
        # initial guess for F
        F0 = np.diag(svS) @ svV
        # initial value for Gama to be updated
        Gama0 = np.zeros((L, self.K))

        # step d: iterate until convergence
        iter, tol = 0, float('inf')
        # ALS estimate
        while iter < MaxIter and tol > MinTol:
            Gama1, F1 = self.als_est(F0, Y0, X0)
            tol_Gamma = abs(Gama1 - Gama0).max()
            tol_F = abs(F1 - F0).max()
            tol = max(tol_Gamma, tol_F)

            if verbose:
                print('iter {}: tolGamma = {} and tolFac = {}'.format(iter, tol_Gamma, tol_F))
            F0, Gama0 = F1, Gama1
            iter += 1

        ## step 2: estimate Gama_tr with treated units before treatment and estimated F1
        Y10, X10 = self._prepare_matrices(self.df.query("tr_group==1 & post_period==0"))
        # dataset demision 
        N_tr, T0, L = X10.shape
        vec_len = L*self.K
        numer, denom = np.zeros(vec_len), np.zeros((vec_len, vec_len))
        for t in range(T0):
            for i in range(N_tr):
                X0_slice = X10[i, t, :] # X_it is Lx1 vector for each i and t
                Ft_slice = F1[:, t] # F_t is Kx1 vector for each t
                # compute kronecker product
                kron_prod = np.kron(X0_slice, Ft_slice)
                # update numer and denom
                numer += kron_prod * Y10[i, t]
                denom += np.outer(kron_prod, kron_prod)

        # solve for gama using matrix left division
        Gama_tr = _mldivide(denom, numer).reshape(L, self.K)

        ## step 3: compute counterfactual for treated units all time periods
        Y1, X1 = self._prepare_matrices(self.df.query("tr_group==1"))
        Y_syn = np.zeros((N_tr, T))
        for i in range(N_tr):
            for t in range(T):
                Y_syn[i, t] = X1[i, t, :]@Gama_tr@F1[:, t]
        # store the result
        self.Y_syn = Y_syn

    def als_est(self, F0, Y0, X0):
        """
        Alternate Least Squares estimation for Gamma and F
        """        
        # dataset dimension
        N_co, T, L = X0.shape
        
        # step b: with F fixed, estimate Gamma, 
        vec_len = L*self.K
        numer, denom = np.zeros(vec_len), np.zeros((vec_len, vec_len))
        for t in range(T):
            for i in range(N_co):
                X0_slice = X0[i, t, :] # X_it is Lx1 vector for each i and t
                Ft_slice = F0[:, t] # F_t is Kx1 vector for each t
                # compute kronecker product
                kron_prod = np.kron(X0_slice, Ft_slice)
                # update numer and denom
                numer += kron_prod * Y0[i, t]
                denom += np.outer(kron_prod, kron_prod)  

        # solve for gama using matrix left division
        Gama1 = _mldivide(denom, numer).reshape(L, self.K)  

        # step c: with Gama fixed, estimate F
        F1 = np.zeros((self.K, T))
        for t in range(T):
            denom = Gama1.T@X0[:,t,:].T@X0[:,t,:]@Gama1
            numer = Gama1.T@X0[:,t,:].T@Y0[:,t]
            F1[:, t] = _mldivide(denom, numer)
        return Gama1, F1

    def _prepare_matrices(self, df):
        Y = df.pivot(index=self.id, columns=self.time, values=self.outcome).values
        X = np.array([df.pivot(index=self.id, columns=self.time, values=x).values for x in self.covariates]).transpose(1, 2, 0)

        return Y, X

    def fit(self, verbose=True, MinTol=1e-6, MaxIter=1e3):
        """
        Fit the model to the data.
        """
        self.run_gsc_ipca2(verbose=verbose, MinTol=MinTol, MaxIter=MaxIter)
        return self