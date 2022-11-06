```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from plotting_tools import df
```

# In this phase we present the most important features for the features of Rating and Installs using the Random Forest and XGBoost algorithms respectively

- [Random Forest](#Random-Forest)
- [XGBoost](#XGBoost)

## Random Forest

### Feature Importance based on Rating
```python
X_train, X_test, y_train, y_test = train_test_split(df.drop('rating', axis = 1), df['rating'], test_size = 0.2)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
best_model = RandomForestRegressor()
best_model = best_model.fit(X_train, y_train)
X = df.drop(['rating'], axis = 1)
feature_importance = best_model.feature_importances_
plt.figure(figsize = (15, 8))
rel_imp = pd.Series(feature_importance, index = X.columns).sort_values(inplace = False)
rel_imp.T.plot(kind = 'barh', color = 'r')
plt.xlabel('Variable Importance')
plt.yticks(fontsize = 11.5, rotation = 45)
plt.gca().legend_ = None
plt.savefig('models_selection_plots/Feature Importance based on Rating')
```
## XGBoost

### Feature Importance based on Installs
```python
X_train, X_test, y_train, y_test = train_test_split(df.drop('installs', axis = 1), df['installs'], test_size = 0.2)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model = XGBRegressor()
model = model.fit(X_train, y_train)
X = df.drop(['installs'], axis = 1)
feature_importance = model.feature_importances_
plt.figure(figsize = (15, 8))
rel_imp = pd.Series(feature_importance, index = X.columns).sort_values(inplace = False)
rel_imp.T.plot(kind = 'barh', color = 'r')
plt.xlabel('Variable Importance')
plt.yticks(fontsize = 11.5, rotation = 45)
plt.gca().legend_ = None
plt.savefig('models_selection_plots/Feature Importance based on Installs')
```
