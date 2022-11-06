import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder 
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# In this step we run the previous functions without the loggers so we can create our plots 

def load_data():
    df = pd.read_csv('Google-Playstore.csv')
    df.rename(lambda x: x.lower().strip().replace(' ', '_'), axis = 1, inplace = True)
    df['category'] = df['category'].str.replace('Music & Audio', 'Music')
    df['category'] = df['category'].str.replace('Educational', 'Education')
    df['size'] = pd.to_numeric(df['size'].str.replace(r'[k, M]', ''), errors = 'coerce', downcast = 'float')
    df['scraped_date'] = pd.to_datetime(df['scraped_time']).dt.date
    df['scraped_time'] = pd.to_datetime(df['scraped_time']).dt.time
    df['released'] = pd.to_datetime(df['released'], format ='%b %d, %Y', 
                                    infer_datetime_format = True, errors = 'coerce')
    df['last_updated'] = pd.to_datetime(df['last_updated'], format ='%b %d, %Y', 
                                        infer_datetime_format = True, errors = 'coerce')
    df['scraped_date'] = pd.to_datetime(df['scraped_date'])
    df.drop(['app_id', 'installs', 'developer_id', 'developer_website', 'developer_email', 'privacy_policy', 
             'editors_choice', 'scraped_time'], axis = 1, inplace = True)
    df['installs'] = (df['minimum_installs'] + df['maximum_installs']) / 2 
    df.drop(['minimum_installs', 'maximum_installs'], axis = 1, inplace = True)
    df.interpolate(method = 'pad', limit_direction = 'forward', inplace = True)
    dp = df.duplicated()
    df = df[~dp]
    df['category'].value_counts(ascending = False)[:10].index.tolist()
    topcat = df[df["category"].isin(['Education', 'Music', 'Tools', 'Business', 'Entertainment', 'Lifestyle', 
                                     'Books & Reference', 'Personalization', 'Health & Fitness', 'Productivity'])]
    return topcat

topcat = load_data()

def preprocess(topcat):
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
    return df

df = preprocess(topcat)

def pca_clustering(df):
    pca = PCA(n_components = 3)
    pca_fit = pca.fit_transform(df)
    kmeans_pca = KMeans(n_clusters = 4, n_init = 10, max_iter = 300, random_state = 42)
    kmeans_pca.fit(pca_fit)
    pca_df = pd.concat([df.reset_index(drop = True), pd.DataFrame(pca_fit)], axis = 1)
    pca_df.columns.values[-3: ] = ['Component 1','Component 2','Component 3']
    pca_df['K-means PCA'] = kmeans_pca.labels_
    pca_df['Clusters'] = pca_df['K-means PCA'].map({0: 'one', 1: 'two', 2: 'three', 3: 'four'})
    return pca_df

pca_df = pca_clustering(df)
