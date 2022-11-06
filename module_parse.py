import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from math import sqrt
from plotting_tools import df

# In this step we use the argparse which enables us to select the model from the models selction function

def models_parse(df, selection=True):
    try:
        X_train, X_test, y_train, y_test = train_test_split(df.drop('rating', axis = 1), df['rating'], test_size = 0.2)
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        results = {}
        if not selection:
           models = [LinearRegression(),
                     Ridge(),
                     DecisionTreeRegressor()]
        else:
           models = [RandomForestRegressor()]
        for model in models:
            fit = model.fit(X_train, y_train)
            y_pred = fit.predict(X_test)
            results[fit.__class__.__name__] = [
                round(r2_score(y_test, y_pred), 2),
                round(mean_absolute_error(y_test, y_pred), 2),
                round(mean_squared_error(y_test, y_pred), 2),
                round(sqrt(mean_squared_error(y_test, y_pred)), 2)]
        index = ['R squared', 'Mean Absolute Error', 'Mean Squared Error', 'Root Mean Squared Error']
        results_parse = pd.DataFrame(data = results, index = index, columns = list(results.keys()))
    except Exception as e:
        return True, str(e)
    if results_parse.shape[0] == 0:
        print('Empty dataframe')
        return True, 'Empty Dataframe'
    return False, results_parse

error, results_parse = models_parse(df)

if error:
    print(f"Error: {results_parse}")
else:
    pass
