import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LassoCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR


def rmse_cv(model, x, y, random_state=17):
    """Root Mean Square Error: error function used by the competition."""
    rmse = np.sqrt(-cross_val_score(
        model, x, y, scoring="neg_mean_squared_error", cv=KFold(10, shuffle=True, random_state=random_state)
    ))
    return rmse


def get_best_model(model, parameters, x, y):
    """Returns the best parameters for the model."""
    grid_obj = GridSearchCV(model, parameters, cv=10, scoring="neg_mean_squared_error")
    grid_obj = grid_obj.fit(x, y)

    # Set the model to the best combination of parameters.
    print(grid_obj.best_params_)
    return grid_obj.best_estimator_


models = dict()
rmse = dict()

# Load training and test data.
train = pd.read_csv("HousePricesTrainClean.csv")
final_test = pd.read_csv("HousePricesTestClean.csv")
# Save target feature.
x = train.drop("SalePrice", axis=1)
y = train['SalePrice']
# Split data into test and train.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=17)

# MODELS


# Lasso model.
models['lasso'] = LassoCV(
    # Search for the optimal alpha in this interval.
    alphas=np.arange(start=0.0005, stop=0.01, step=0.0001),
    cv=KFold(10, shuffle=True, random_state=17)
).fit(x_train, y_train)
rmse['lasso'] = rmse_cv(models['lasso'], x_train, y_train)

print("Lasso Regression (10-fold cross-validation)")
print("\tRMSE=%.3f for Alpha=%.3f" % (rmse['lasso'].mean(), models['lasso'].alpha_))

# GBR model.
models['gbr'] = GradientBoostingRegressor(n_estimators=200, min_samples_leaf=3, max_features=0.1, learning_rate=0.1,
                                          max_depth=3)
# Choose some parameter combinations to try.
parameters = {
    # 'learning_rate': [0.2, 0.1, 0.05],
    # 'max_depth': [3]
}
# Get best combinations of parameters.
models['gbr'] = get_best_model(models['gbr'], parameters, x_train, y_train)

# Fit the best algorithm to the data.
models['gbr'].fit(x_train, y_train)
rmse['gbr'] = rmse_cv(models['gbr'], x_train, y_train)

print("Gradient Boosting Regression (10-fold cross-validation)")
print("\tRMSE=%.3f" % (rmse['gbr'].mean()))

# Random Forest model.
models['rf'] = RandomForestRegressor(n_estimators=175, max_depth=11)
# Choose some parameter combinations to try.
parameters = {
    # 'n_estimators': [150, 175],
    # 'max_depth': np.arange(9, 12)
}

# Get best combinations of parameters.
models['rf'] = get_best_model(models['rf'], parameters, x_train, y_train)
models['rf'] = models['rf'].fit(x_train, y_train)
rmse['rf'] = rmse_cv(models['rf'], x_train, y_train)

print("Random Forest Regression (10-fold cross-validation)")
print("\tRMSE=%.3f" % (rmse['rf'].mean()))

# print(pd.DataFrame({
#    'Variable': x_train.columns,
#    'Importance': np.round(models['rf'].feature_importances_, 4)
# }).sort_values('Importance', ascending=False))


# Support Vector Machines model.
models['svm'] = SVR(gamma='auto', C=1.1)
# Choose some parameter combinations to try.
parameters = {
    # 'C': [0.8, 0.9, 1, 1.1, 1.2],
    # 'gamma': ['auto', 'scale']
}
# Get best combinations of parameters.
models['svm'] = get_best_model(models['svm'], parameters, x_train, y_train)
models['svm'] = models['svm'].fit(x_train, y_train)
rmse['svm'] = rmse_cv(models['svm'], x_train, y_train)

print("Support Vector Machines (10-fold cross-validation)")
print("\tRMSE=%.3f" % (rmse['svm'].mean()))


# AdaBoost model.
models['ada'] = AdaBoostRegressor()
# Choose some parameter combinations to try.
parameters = {
    'n_estimators': [150, 200, 250]
}

# Get best combinations of parameters.
models['ada'] = get_best_model(models['ada'], parameters, x_train, y_train)
models['ada'] = models['ada'].fit(x_train, y_train)
rmse['ada'] = rmse_cv(models['ada'], x_train, y_train)

print("AdaBoost (10-fold cross-validation)")
print("\tRMSE=%.3f" % (rmse['ada'].mean()))


# Compare training and testing accuracies.
for name in rmse.keys():
    print("%s" % (name.upper()))
    print("\tRMSE: %.3f" % (rmse[name].mean()))
    predictions = models[name].predict(x_test)
    print("\tTest: %.3f" % (np.sqrt(mean_squared_error(y_test, predictions))))


# Prediction.
model_lasso = Lasso(alpha=0.001).fit(x, y)
y_pred = model_lasso.predict(final_test)

# Create submission.
submission = pd.DataFrame({
    "Id": final_test.Id,
    "SalePrice": y_pred
})
submission.to_csv('submission.csv', index=False)
