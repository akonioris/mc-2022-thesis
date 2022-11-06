import pandas as pd
import time
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# This step provides
# Label encoding
# Scaling

def preprocess(topcat, logger):
    start = time.time()
    try:
        df_encoded = topcat.applymap(lambda x: 1 if x == True else x)
        df = df_encoded.applymap(lambda x: 0 if x == False else x)
        obj_column = df_encoded.dtypes[topcat.dtypes == 'object'].index
        labelencoder_X = LabelEncoder()
        for column in obj_column:
            df_encoded[column] = labelencoder_X.fit_transform(df_encoded[column])
        df_encoded.drop(['last_updated', 'released', 'scraped_date'], axis = 1, inplace = True)
        scaler = MinMaxScaler() 
        scaler.fit(df_encoded)
        df = pd.DataFrame(scaler.transform(df_encoded), index = df_encoded.index, columns = df_encoded.columns)
    except Exception as e:
        return True, str(e)
        
    if df.shape[0] == 0:
        logger.error('Empty dataframe')
        return True, 'Empty Dataframe'

    end = time.time()
    logger.info(f"Execution time of preprocessing function is: {end - start} seconds")
    return df


