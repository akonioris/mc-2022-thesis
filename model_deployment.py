import pandas as pd
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from math import sqrt

# In this phase, a Random Forest regression model is performed, which gave us the best accuracy from the previous step

def best_model(df, logger):
    start = time.time()
    try:
        iso = df[['rating_count', 'app_name', 'size', 'rating']]
        X_train, X_test, y_train, y_test = train_test_split(iso.drop('rating', axis = 1), iso['rating'], test_size = 0.2)
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        best_model = RandomForestRegressor()
        results = {}
        fit = best_model.fit(X_train, y_train)
        y_pred = fit.predict(X_test)
        results[fit.__class__.__name__] = [
            round(r2_score(y_test, y_pred), 2),
            round(mean_absolute_error(y_test, y_pred), 2),
            round(mean_squared_error(y_test, y_pred), 2),
            round(sqrt(mean_squared_error(y_test, y_pred)), 2)]
        index = ['R squared', 'Mean Absolute Error', 'Mean Squared Error', 'Root Mean Squared Error']
        outcome = pd.DataFrame(data = results, index = index, columns = list(results.keys()))
    except Exception as e:
        return True, str(e)
        
    if outcome.shape[0] == 0:
        logger.error('Empty dataframe')
        return True, 'Empty Dataframe'

    logger.info(f"Result of best_model's function is: \n{outcome}")
    end = time.time()
    logger.info(f"Execution time of best_model's function is: {end - start} seconds")
    return outcome
