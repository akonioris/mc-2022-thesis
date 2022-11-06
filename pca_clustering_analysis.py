import pandas as pd
import time
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# This phase provides the dimensionality reduction of the data using PCA and K-means Clustering based 
# With 3 principal components and 4 clusters based on the results of the pca_clustering_plots module

def pca_clustering(df, logger):
    start = time.time()
    try:
        pca = PCA(n_components = 3)
        pca_fit = pca.fit_transform(df)
        kmeans_pca = KMeans(n_clusters = 4, n_init = 10, max_iter = 300, random_state = 42)
        kmeans_pca.fit(pca_fit)
        pca_df = pd.concat([df.reset_index(drop = True), pd.DataFrame(pca_fit)], axis = 1)
        pca_df.columns.values[-3: ] = ['Component 1','Component 2','Component 3']
        pca_df['K-means PCA'] = kmeans_pca.labels_
        pca_df['Clusters'] = pca_df['K-means PCA'].map({0: 'one', 1: 'two', 2: 'three', 3: 'four'})
    except Exception as e:
        return True, str(e)
        
    if pca_df.shape[0] == 0:
        logger.error('Empty dataframe')
        return True, 'Empty Dataframe'

    end = time.time()
    logger.info(f"Execution time of pca_clustering function is: {end - start} seconds")
    return pca_df
