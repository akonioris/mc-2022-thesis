import pandas as pd
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from math import sqrt


# This step provides the best model to predict rating among the features

def models(df, logger):
    start = time.time()
    try:
        X_train, X_test, y_train, y_test = train_test_split(df.drop('rating', axis = 1), df['rating'], test_size = 0.2)
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        results = {}
        models = [LinearRegression(),
                  Ridge(),
                  DecisionTreeRegressor(),
                  RandomForestRegressor()]
        for model in models:
            fit = model.fit(X_train, y_train)
            y_pred = fit.predict(X_test)
            results[fit.__class__.__name__] = [
                round(r2_score(y_test, y_pred), 2),
                round(mean_absolute_error(y_test, y_pred), 2),
                round(mean_squared_error(y_test, y_pred), 2),
                round(sqrt(mean_squared_error(y_test, y_pred)), 2)]
        index = ['R squared', 'Mean Absolute Error', 'Mean Squared Error', 'Root Mean Squared Error']
        results_df = pd.DataFrame(data = results, index = index, columns = list(results.keys()))
    except Exception as e:
        return True, str(e)
        
    if results_df.shape[0] == 0:
        logger.error('Empty dataframe')
        return True, 'Empty Dataframe'
    
    logger.info(f"Results of models' function are: \n{results_df}")
    end = time.time()
    logger.info(f"Execution time of models' function is: {end - start} seconds")
    return results_df
