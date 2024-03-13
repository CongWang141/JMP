import numpy as np
import scipy.linalg as sla
import scipy.sparse.linalg as ssla

# matrix left/right division (following MATLAB function naming)
_mldivide = lambda denom, numer: sla.lstsq(np.array(denom), np.array(numer))[0]
_mrdivide = lambda numer, denom: (sla.lstsq(np.array(denom).T, np.array(numer).T)[0]).T

class gsc_ipca(object):
    def __init__(self) -> None:
        pass

    def fit(self, df, id, time, outcome, covariates, K, MaxIter=100, tol=1e-6, verbose=True):
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

        # gen Y and X to estimate F and Gama
        Y, X = self._prepare_matrix(df, covariates, id, time, outcome)
        _, _, L = X.shape

        # initial guess for F0 and Gama0
        svU, svS, svV = ssla.svds(Y, k=K)
        # reverse the order of singular values and vectors
        svU, svS, svV = svU[:, ::-1], svS[::-1], svV[::-1, :]
        # initial guess for F0
        F0 = np.diag(svS) @ svV
        # initial guess for Gama0
        Gama0 = np.zeros((L, K))

        # estimate F1 and Gama1 by ALS algorithm
        iter, tol = 0, float('inf')
        while iter < MaxIter and tol > 1e-6:
            Gama1, F1 = self.als_est(F0, Y, X, K)
            tol_Gama = abs(Gama1 - Gama0).max()
            tol_F = abs(F1 - F0).max()
            tol = max(tol_Gama, tol_F)

            if verbose:
                print('iter {}: tol_Gama: {}, tol_F: {}'.format(iter, tol_Gama, tol_F))
            F0, Gama0 = F1, Gama1
            iter += 1

        # store the estimated F and Gama
        self.F = F1
        self.Gama = Gama1

    def als_est(self, F0, Y, X, K):
        """
        Alternating Least Squares (ALS) algorithm to estimate F1 and Gama1
        """
        # dataset dimension
        N, T, L = X.shape

        # with F0 fixed, estimate Gama1
        vec_len = L * K
        numer, demon = np.zeros(vec_len), np.zeros((vec_len, vec_len))
        for t in range(T):
            for i in range(N):
                # x_it is Lx1 vector for each unit i at time t
                X_slice = X[i, t, :]
                # F_t is Kx1 vector for each time t
                F_slice = F0[:, t]
                # compute the kronecker product of F_t and x_it
                kron_prod = np.kron(X_slice, F_slice)
                # update numer and demon
                numer += kron_prod * Y[i, t]
                demon += np.outer(kron_prod, kron_prod)
        # solve for Gama1 using matrix left division
        Gama1 = _mldivide(demon, numer).reshape(L, K)

        # with Gama1 fixed, estimate F1
        F1 = np.zeros((K, T))
        for t in range(T):
            denom = Gama1.T@X[:, t, :].T@X[:, t, :]@Gama1
            numer = Gama1.T@X[:, t, :].T@Y[:, t]
            F1[:, t] = _mldivide(denom, numer)
        return Gama1, F1

    def _prepare_matrix(self, df, covariates, id, time, outcome):
        Y = df.pivot(index=id, columns=time, values=outcome).values
        X = np.array([df.pivot(index=id, columns=time, values=x).values for x in covariates]).transpose(1, 2, 0)    
        return Y, X
    
    def predict(self, df, id, time, outcome, covariates, K, treated):
        """
        Predict the counterfactual outcome for treated units
        """
        # if treated is not None, estimate Gama using treated units across all time periods
        # (convinient for conformal inference)
        if treated:
            df_pre = df[df[treated] == 0]
            Y, X = self._prepare_matrix(df_pre, covariates, id, time, outcome)
        else:
            Y, X = self._prepare_matrix(df, covariates, id, time, outcome)
        # gen Y and X to estimate Gama for treated units
        N, T, L = X.shape

        # estimate Gama for treated units
        vec_len = L * K
        numer, demon = np.zeros(vec_len), np.zeros((vec_len, vec_len))
        for t in range(T):
            for i in range(N):
                X_slice = X[i, t, :]
                F_slice = self.F[:, t]
                kron_prod = np.kron(X_slice, F_slice)
                numer += kron_prod * Y[i, t]
                demon += np.outer(kron_prod, kron_prod)
        # solve for Gama using matrix left division
        Gama1 = _mldivide(demon, numer).reshape(L, K)

        # compute counterfactual for treated units all time periods
        Y, X = self._prepare_matrix(df, covariates, id, time, outcome)
        N, T, L = X.shape
        Y_syn = np.zeros((N, T))
        for i in range(N):
            for t in range(T):
                Y_syn[i, t] = X[i, t, :] @ Gama1 @ self.F[:, t]
                
        return Y_syn

        

