"""
Univariate missing value imputation with SimpleImputer class

Code readapted from Scikit-Learn documentation
"""

import numpy as np
from sklearn.impute import SimpleImputer

imp = SimpleImputer(missing_values=np.nan, strategy='mean')

imp.fit([[1, 2], [np.nan, 3], [7, 6]])

SimpleImputer(add_indicator=False, copy=True, fill_value=None,
              missing_values=nan, strategy='mean', verbose=0)

X = [[np.nan, 2], [6, np.nan], [7, 6]]

imp.transform(X)


"""
Multivariate missing value imputation with IterativeImputer class

Code readapted from Scikit-Learn documentation
"""

import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

imp = IterativeImputer(max_iter=10, random_state=0)
imp.fit([[1, 2], [3, 6], [4, 8], [np.nan, 3], [7, np.nan]])

IterativeImputer(add_indicator=False, estimator=None,
                 imputation_order='ascending', initial_strategy='mean',
                 max_iter=10, max_value=None, min_value=None,
                 missing_values=nan, n_nearest_features=None,
                 random_state=0, sample_posterior=False, tol=0.001,
                 verbose=0)

X_test = [[np.nan, 2], [6, np.nan], [np.nan, 6]]

np.round(imp.transform(X_test))

