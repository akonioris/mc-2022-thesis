```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from plotting_tools import df
```

# This phase presents the graphs that have been produced by the Principal Component Analysis and the K-means Clustering algorithms in order to keep the optimal number of principal components and clusters respectively

- [PCA](#PCA)
- [K-means Clustering](#K--means-Clustering)

## PCA

### Explained Variance by Principal Components

```python
pca = PCA()
pca.fit_transform(df)
plt.figure(figsize = (15, 8))
plt.plot(range(1, 14), pca.explained_variance_ratio_.cumsum(), marker = 'o', linestyle = '--')
plt.title('Explained Variance by Principal Components')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.savefig('pca_clustering_plots/Explained Variance by Principal Components')
```
## K-means Clustering

### Distribution of Clusters based on SSE
```python
pca = PCA(n_components = 3)
pca_fit = pca.fit_transform(df)
kmeans_kwargs = {"init": "random", "n_init": 10, "max_iter": 300, "random_state": 42}
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters = k, **kmeans_kwargs)
    kmeans.fit(pca_fit)
    sse.append(kmeans.inertia_)
plt.figure(figsize = (15, 8))
plt.plot(range(1, 11), sse)
plt.xticks(range(1, 11))
plt.title('Distribution of Clusters based on SSE')
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
plt.savefig('pca_clustering_plots/Distribution of Clusters based on SSE')
```
