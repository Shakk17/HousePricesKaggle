import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import skew
from scipy.stats import norm
from scipy.stats.stats import pearsonr

from sklearn.preprocessing import LabelEncoder, OneHotEncoder


# LOADING DATASETS.

# Load train and test data.
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
# Merge datasets.
data = pd.concat(objs=[train, test], axis=0, ignore_index=True, sort=True)
# Drop target variable.
data = data.drop(["Id", "SalePrice"], axis=1)

# DATA ANALYSIS

# We try to get some info about the data.
cat_var_mask = data.dtypes == object
cat_var = data.columns[cat_var_mask]
num_var = data.columns[~cat_var_mask]
print("%d Categorical Variables\n%d Numerical Variables\n" % (len(cat_var), len(num_var)))
# Describing numerical features.
print(train.describe(), end="\n\n")
# Describing categorical features.
print(train.describe(exclude=[np.number]), end="\n\n")

# Checking the distribution of the target feature.
matplotlib.rcParams['figure.figsize'] = (8.0, 6.0)
# Fit the data with a normal distribution.
sns.distplot(train['SalePrice'], fit=norm)
(mu, sigma) = norm.fit(train['SalePrice'])
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')
plt.show()

# SalePrice has a skewed distribution. Let's normalize it.
train["SalePrice"] = np.log1p(train["SalePrice"])
(mu, sigma) = norm.fit(train['SalePrice'])

matplotlib.rcParams['figure.figsize'] = (8.0, 6.0)
sns.distplot(train['SalePrice'] , fit=norm)
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')
plt.show()

# FIXING MISSING VALUES

# How many missing values?
missing = data.isnull().sum().sort_values(ascending=False)
percent = (data.isnull().sum() / len(data)).sort_values(ascending=False) * 100
missing_data = pd.concat([missing, percent], axis=1, keys=['Missing', '%'])
missing_data = missing_data[missing_data["%"] != 0]
print(missing_data)

# Filling missing values by following the documentation.
fill_no = ["PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu", "GarageType", "GarageFinish", "GarageQual",
           "GarageCond",
           "BsmtCond", "BsmtExposure", "BsmtQual", "BsmtFinType1", "BsmtFinType2", "MasVnrType"]
for var in fill_no:
    data[var] = data[var].fillna("None")

fill_mode = ["GarageYrBlt", "MSZoning", "Functional", "Exterior1st", "Exterior2nd", "SaleType", "Electrical",
             "KitchenQual"]
for var in fill_mode:
    data[var] = data[var].fillna(data[var].mode()[0])

fill_zero = ["MasVnrArea", "BsmtFullBath", "BsmtHalfBath", "GarageCars", "GarageArea", "TotalBsmtSF",
             "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF"]
for var in fill_zero:
    data[var] = data[var].fillna(0)

data["LotFrontage"] = data.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))
data = data.drop(['Utilities'], axis=1)

# How many missing values?
missing = data.isnull().sum().sort_values(ascending=False)
percent = (data.isnull().sum() / len(data)).sort_values(ascending=False) * 100
missing_data = pd.concat([missing, percent], axis=1, keys=['Missing', '%'])
missing_data = missing_data[missing_data["%"] != 0]
print(missing_data)


# FEATURE CORRECTION

# Apply log1p function to all the skewed variables over a certain threshold (0.75).
skewed_feats = data[num_var].apply(lambda x: skew(x.dropna()))
skewness = pd.DataFrame({"Skewness": skewed_feats}).sort_values(by='Skewness', ascending=False)
print(skewness)

skewed_feats = skewed_feats[skewed_feats > 0.75]
data[skewed_feats.index] = np.log1p(data[skewed_feats.index])

skewed_feats = data[num_var].apply(lambda x: skew(x.dropna()))
skewness = pd.DataFrame({"Skewness": skewed_feats}).sort_values(by='Skewness', ascending=False)
print(skewness)

# CORRELATION ANALYSIS

# Correlation between all the features.
corr_mat = train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corr_mat, square=True, cmap="Blues")
plt.show()

# SalePrice correlation matrix
cols = corr_mat.nlargest(n=10, columns='SalePrice').index
cm = np.corrcoef(train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, cmap="Blues",
                 yticklabels=cols.values, xticklabels=cols.values)
plt.show()

# OneHotEncoding for all the categorical variables.
print("Number of Variables before OHE: "+str(data.shape[1]))
data = pd.get_dummies(data)
print("Number of Variables after OHE: "+str(data.shape[1]))

# SAVING OF PREPROCESSED DATA

X_train = data[:train.shape[0]].copy()
X_train['SalePrice'] = train.SalePrice
X_train.to_csv("HousePricesTrainClean.csv")

X_test = data[train.shape[0]:]
X_test['Id'] = test.Id
X_test.to_csv("HousePricesTestClean.csv")
