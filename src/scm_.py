import numpy as np
from scipy.optimize import minimize, Bounds, LinearConstraint
from sklearn.utils.validation import check_X_y
class scm(object):
    def __init__(self, df, id, time, outcome, treated, v):
        """
        df: pandas dataframe, panel data
        id: string, name of the id column
        time: string, name of the time column
        outcome: string, name of the outcome column
        treated: string, name of the treatment column
        v: weigting matrix v, to assign weight to observations
        """
        # initialize the scm object with given data
        self.df = df
        self.id = id
        self.time = time
        self.outcome = outcome
        self.treated = treated
        self.v = v

        # create post_period and tr_group columns
        self.df['post_period'] = self.df.groupby(self.time)[self.treated].transform('max')
        self.df['tr_group'] = self.df.groupby(self.id)[self.treated].transform('max')

    def run_scm(self):
        """
        Run the synthetic control method to estimate the counterfactual.
        """
        # control group pre-treatment period
        Y00 = self.df.query('post_period == 0 & tr_group == 0').pivot(index=self.time, columns=self.id, values=self.outcome).values
        # treated group pre-treatment period
        Y10 = self.df.query('post_period == 0 & tr_group == 1').pivot(index=self.time, columns=self.id, values=self.outcome).mean(axis=1)

        # Initial guess for the weights: could be 0, uniform or based on some other logic
        initial_w = (np.ones(Y00.shape[1])/Y00.shape[1])

        if self.v is None:
            # If no weights are given, use the default weights
            v = np.diag(np.ones(Y00.shape[0])/Y00.shape[0])
        
        X, y = check_X_y(Y00, Y10)
        # Solve for the weights
        weights = self.solve_weights(X, y, initial_w, v)

        # compute Y_syn
        Y0 = self.df.query('tr_group == 0').pivot(index=self.time, columns=self.id, values=self.outcome).values

        Y_syn = Y0 @ weights

        # store the results
        self.Y_syn = Y_syn
    def solve_weights(self, X, y, initial_w, v):
        """
        Solve for the weights using the given data and initial weights.
        """
        # Define the objective function to minimize: the sum of squares of the residuals
        def fun_obj(w, X, y, v):
            return np.mean(np.sqrt((y - X @ w).T @ v @ (y - X @ w)))
    
        # Define the constraints: the weights should sum to 1
        constraints = LinearConstraint(np.ones(X.shape[1]), lb= 1, ub= 1)

        # Define the bounds: the weights should be between 0 and 1
        bounds = Bounds(lb=0, ub=1)

        # Use the SLSQP method which supports both bounds and constraints
        result = minimize(fun_obj, x0=initial_w, args=(X, y, v), method='SLSQP', bounds=bounds, constraints=constraints)
        return result.x
    
    def fit(self):
        """
        Fit the model to the data.
        """
        self.run_scm()
        return self




