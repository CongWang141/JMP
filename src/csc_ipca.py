import numpy as np
import pandas as pd
import scipy.linalg as sla
import scipy.sparse.linalg as ssla

# matrix left/right division (following MATLAB function naming)
_mldivide = lambda denom, numer: sla.lstsq(np.array(denom), np.array(numer))[0]
_mrdivide = lambda numer, denom: (sla.lstsq(np.array(denom).T, np.array(numer).T)[0]).T

class CSC_IPCA(object):
    def __init__(self) -> None:
        pass

    def fit(self, df, id, time, outcome, treated, covariates, K, MaxIter=100, MinTol=1e-6, verbose=False):
        """
        df: pd.DataFrame, should be the control data
        id: str, column name for unit id
        time: str, column name for time period
        outcome: str, column name for outcome variable
        covariates: list of str, column names for covariates
        treated: str, column name for treated unit
        K: int, number of factors
        L: int, number of covariates
        """
        # Initializes the model with given parameters and data.
        self.df = df
        self.id = id
        self.time = time
        self.outcome = outcome
        self.covariates = covariates
        self.treated = treated
        self.K = K 

        # gen tr_group and post_period
        df['tr_group'] = df.groupby(id)[treated].transform('max')
        df['post_period'] = df.groupby(time)[treated].transform('max')

        # gen Y and X to estimate F and Gama
        Y0, X0 = _prepare_matrix(df.query("tr_group==0"), covariates, id, time, outcome)
        _, _, L = X0.shape

        # initial guess for F0 and Gama0
        svU, svS, svV = ssla.svds(Y0, k=K)
        # reverse the order of singular values and vectors
        svU, svS, svV = np.fliplr(svU), svS[::-1], np.flipud(svV)
        # initial guess for F0
        F0 = np.diag(svS) @ svV
        # initial guess for Gama0
        Gama0 = np.zeros((L, K))

        # estimate F1 and Gama1 by ALS algorithm
        iter, tol = 0, float('inf')
        while iter < MaxIter and tol > MinTol:
            Gama1, F1 = als_est(F0, Y0, X0, K)
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
    
    def predict(self):
        """
        Predict the counterfactual outcome for treated units
        df: dataframe, should be the treated data
        """

        # gen Y and X from treated data before treatment to estimate Gama_tr
        Y, X = _prepare_matrix(self.df.query("tr_group==1 & post_period==0"), self.covariates, self.id, self.time, self.outcome)

        # estimate Gama_tr for treated units
        Gama_tr = estimate_gama(Y, X, self.F, self.K)

        # compute counterfactual for treated units all time periods
        _, X = _prepare_matrix(self.df.query("tr_group==1"), self.covariates, self.id, self.time, self.outcome)
        Y_syn = compute_syn(X, Gama_tr, self.F)
        self.Gama_tr = Gama_tr
                
        return Y_syn
    
    def inference(self, nulls, window, verbose=False, MaxIter=100, MinTol=1e-6):
        """
        Conduct confermal inference
        """
        # gen the treatment starting time
        tr_start = self.df.query("post_period==1")[self.time].min()
        # gen a data frame to store the computed confidence interval
        CI_df = pd.DataFrame()
        for t in range(tr_start, tr_start + window):
            # append the target year to the control data
            df_app = self.df[(self.df[self.time] < tr_start) | (self.df[self.time]==t)]
            # gen a dic to store the result
            p_values = {}
            for null in nulls:
                # remember we use the whole data to estimate Gama and F (different from estimation)
                # assign the null
                null_data = under_null(df_app, null, self.outcome, self.treated)

                # gen Y0, X0 to estimate F
                Y0, X0 = _prepare_matrix(null_data, self.covariates, self.id, self.time, self.outcome)
                _, _, L = X0.shape
                # initialize F0, Gama0
                svU, svS, svV = ssla.svds(Y0, self.K)
                svU, svS, svV = np.fliplr(svU), svS[::-1], np.flipud(svV)
                F0 = np.diag(svS) @ svV
                Gama0 = np.zeros((L, self.K))

                # update F and Gama
                tol, iter = float('inf'), 0
                while tol > MinTol and iter < MaxIter:
                    Gama1, F1 = als_est(F0, Y0, X0, self.K)
                    tol_Gama = abs(Gama1 - Gama0).max()
                    tol_F = abs(F1 - F0).max()
                    tol = max(tol_Gama, tol_F)
                    F0, Gama0 = F1, Gama1
                    iter += 1

                # prepare treated data for the whole period
                Y1, X1 = _prepare_matrix(null_data.query("tr_group==1"), self.covariates, self.id, self.time, self.outcome)
                # compute Gama_tr with treated data
                Gama_tr = estimate_gama(Y1, X1, F1, self.K)
                # compute the Y_syn
                Y_syn = compute_syn(X1, Gama_tr, F1)

                # compute the residual
                residuals = Y1 - Y_syn

                resid_df = pd.DataFrame({
                    'y': Y1.mean(axis=0),
                    'y_hat': Y_syn.mean(axis=0),
                    'residuals': residuals.mean(axis=0),
                    'post_period': null_data.groupby(self.time).post_period.max()
                })[lambda d: d.index < (tr_start + window)]

                # compute p value
                p_val = p_value(resid_df, 1)
                # append the computed p value to the dic
                p_values[null] = p_val
            # convert into dataframe
            p_values_df = pd.DataFrame(p_values, index=[t]).T

            ci = confidence_interval(p_values_df, 0.1)
            CI_df = pd.concat([CI_df, ci], axis=0)
            if verbose:
                print(f'Compute CI for year {t} is completed!')
        return CI_df

##############################################
# define a function to prepare matrix
def _prepare_matrix(df, covariates, id, time, outcome):
    Y = df.pivot(index=id, columns=time, values=outcome).values
    X = np.array([df.pivot(index=id, columns=time, values=x).values for x in covariates]).transpose(1, 2, 0)    
    return Y, X

# define a function to alteratively solve for gamma and F
def als_est(F0, Y, X, K):
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

# define a function to compute Gama for treated units
def estimate_gama(Y, X, F1, K):
    N, T, L = X.shape
    # with F fixed, estimate Gama
    vec_len = L*K
    numer, denom = np.zeros(vec_len), np.zeros((vec_len, vec_len))
    for t in range(T):
        for i in range(N):
            X_slice = X[i, t, :]
            F_slice = F1[:, t]
            kron_prod = np.kron(X_slice, F_slice)
            # update numer and denom
            numer += kron_prod * Y[i, t]
            denom += np.outer(kron_prod, kron_prod)
    # solve for Gama
    Gama1 = _mldivide(denom, numer).reshape(L, K)
    return Gama1

# define a function to compute counterfactual for treated units
def compute_syn(X, Gama, F):
    N, T, L = X.shape
    Y_syn = np.zeros((N, T))
    for i in range(N):
        for t in range(T):
            Y_syn[i, t] = X[i, t, :] @ Gama @ F[:, t]
    return Y_syn

# define a function to assign the null
def under_null(df, null, outcome, treated):
    data = df.copy()
    y = np.where(data[treated]==1, data[outcome] - null, data[outcome])
    return data.assign(**{outcome: y})

# define a function to compute test statistics
def test_statistic(u_hat, q=1, axis=0):
    return (np.abs(u_hat)**q).mean(axis=axis)**(1/q)

# define a function to compute p value
def p_value(resid_df, q=1):
    u = resid_df.residuals.values
    post_period = resid_df.post_period == 1

    block_permutation = np.stack([np.roll(u, permutations, axis=0)[post_period] for permutations in range(len(u))])

    statistics = test_statistic(block_permutation, q=q, axis=1)
    p_value = np.mean(statistics >= statistics[0])
    return p_value

# define a function to compute confidence interval
def confidence_interval(p_vals, alpha=0.1):
    big_p_vals = p_vals[p_vals.values >= alpha]
    return pd.DataFrame({
        f"{int(100-alpha*100)}_ci_lower": big_p_vals.index.min(),
        f"{int(100-alpha*100)}_ci_upper": big_p_vals.index.max()
    }, index=[p_vals.columns[0]])