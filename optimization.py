import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from math import sqrt
from sklearn.model_selection import train_test_split

# This step provides the optimization function of our analysis

def best_model_optimized(df, logger):
    start = time.time()
    try:
        iso = df[['rating_count', 'app_name', 'size', 'rating']]
        npa = iso.to_numpy()
        X_train, X_test, y_train, y_test = train_test_split(np.delete(npa, 3, axis = 1), npa[:, [3]], test_size = 0.2)
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        best_model = XGBRegressor()
        results = {}
        fit = best_model.fit(X_train, y_train)
        y_pred = fit.predict(X_test)
        results[fit.__class__.__name__] = [
            round(r2_score(y_test, y_pred), 2),
            round(mean_absolute_error(y_test, y_pred), 2),
            round(mean_squared_error(y_test, y_pred), 2),
            round(sqrt(mean_squared_error(y_test, y_pred)), 2)]
        index = ['R squared', 'Mean Absolute Error', 'Mean Squared Error', 'Root Mean Squared Error']
        result_optimized = pd.DataFrame(data = results, index = index, columns = list(results.keys()))
    except Exception as e:
        return True, str(e)
    if result_optimized.shape[0] == 0:
        logger.error('Empty dataframe')
        return True, 'Empty Dataframe'

    logger.info(f"Result of best_model's optimization function is: \n{result_optimized}")
    end = time.time()
    logger.info(f"Execution time of best_model's optimization function is: {end - start} seconds")
    return result_optimized
