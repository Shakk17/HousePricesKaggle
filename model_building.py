import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib

import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import skew
from scipy.stats import norm
from scipy.stats.stats import pearsonr

from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, ElasticNet, Lasso, LassoCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

# Load training and test data.
train = pd.read_csv("HousePricesTrainClean.csv")
X_test = pd.read_csv("HousePricesTestClean.csv")
X_train = train.drop("SalePrice", axis=1)
# Save target feature.
y = train['SalePrice']


def r2_cv(model, X_train, y, random_state=12345678):
    r2 = cross_val_score(model, X_train, y, scoring="r2", cv=KFold(10, shuffle=True, random_state=random_state))
    return r2


def rmse_cv(model, X_train, y, random_state=12345678):
    rmse = np.sqrt(-cross_val_score(
        model, X_train, y, scoring="neg_mean_squared_error", cv=KFold(10, shuffle=True, random_state=random_state)
    ))
    return rmse


model_simple = LinearRegression()
model_simple.fit(X_train, y)
yp = model_simple.predict(X_train)

# Compute R2 for training data and using crossvalidation.
r2_simple_train = r2_score(y, yp)
r2_xval_simple = r2_cv(model_simple, X_train, y)

# Compute RMSE for training data and using crossvalidation.
rmse_simple_train = mean_squared_error(y, yp, multioutput='raw_values')
rmse_xval_simple = rmse_cv(model_simple, X_train, y)

print("Linear Regression")
print("==================================================")
print("\t                  Train R2=%.3f" % r2_simple_train)
print("\t10-fold Crossvalidation R2=%.3f" % (r2_xval_simple.mean()))
print("\t                  Train RMSE=%.3f" % rmse_simple_train)
print("\t10-fold Crossvalidation RMSE=%.3f" % (rmse_xval_simple.mean()))

# Now we try Ridge (L_2) and Lasso (L_1) Regression, with cross-validation.
model_ridge = RidgeCV(
    alphas=[0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75],
    cv=KFold(10, shuffle=True, random_state=12345678)
).fit(X_train, y)
model_lasso = LassoCV(
    alphas=[1, 0.1, 0.001, 0.0005],
    cv=KFold(10, shuffle=True, random_state=12345678)
).fit(X_train, y)

# Create submission.
submission = pd.DataFrame({
        "Id": X_test.Id,
        # "SalePrice": pred_y
    })
submission.to_csv('submission.csv', index=False)

