```python
import matplotlib.pyplot as plt
import seaborn as sns
from plotting_tools import pca_df
```

# This phase presents the analysis that has been made on the features of our data based on the results of the Principal Component Analysis and the K-means Clustering algorithms

- [Clusters by PCA Components](#Clusters-by-PCA-Components)
- [Comparison of Clusters Based on Installs](#Comparison-of-Clusters-Based-on-Installs)
- [Comparison of Clusters Based on Price](#Comparison-of-Clusters-Based-on-Price)
- [Comparison of Clusters Based on Rating](#Comparison-of-Clusters-Based-on-Rating)
- [Comparison of Clusters Among the Categories](#Comparison-of-Clusters-Among-the-Categories)

## Clusters by PCA Components
```python
plt.figure(figsize = (15, 8))
sns.scatterplot(pca_df['Component 1'], pca_df['Component 2'], hue = pca_df['Clusters'])
plt.title('Clusters by PCA Components')
plt.savefig('pca_clustering_plots/Clusters by PCA Components')
```
## Comparison of Clusters Based on Installs
```python
fig, ax = plt.subplots(figsize = [15, 8])
sns.barplot(x = 'Clusters', y ='installs', data = pca_df, palette = 'rainbow')
ax.set(title = 'Comparison of Clusters Based on Installs',
       xlabel ='', ylabel = 'Installs')
ax.set_xticklabels(ax.get_xticklabels(), rotation = 10)
plt.savefig('pca_clustering_plots/Comparison of Clusters Based on Installs')
```
## Comparison of Clusters Based on Price
```python
fig, ax = plt.subplots(figsize = [15, 8])
sns.barplot(x = 'Clusters', y ='price', data = pca_df, palette = 'rainbow')
ax.set(title = 'Comparison of Clusters Based on Price',
       xlabel ='', ylabel = 'Price')
ax.set_xticklabels(ax.get_xticklabels(), rotation = 10)
plt.savefig('pca_clustering_plots/Comparison of Clusters Based on Price')
```
## Comparison of Clusters Based on Rating
```python
fig, ax = plt.subplots(figsize = [15, 8])
sns.boxplot(x = 'Clusters', y ='rating', data = pca_df, palette = 'rainbow')
ax.set(title = 'Comparison of Clusters Based on Rating',
       xlabel ='', ylabel = 'Rating')
ax.set_xticklabels(ax.get_xticklabels(), rotation = 10)
plt.savefig('pca_clustering_plots/Comparison of Clusters Based on Rating')
```
## Comparison of Clusters Among the Categories
```python
exe = pca_df[['category', 'Clusters']]
exe.replace([2/9, 6/9, 1, 1/9, 3/9, 5/9, 0, 0.7777777777777777, 4/9, 8/9], 
               ['Education', 'Music', 'Tools', 'Business', 'Entertainment', 'Lifestyle', 
                'Books & Reference', 'Personalization', 'Health & Fitness', 'Productivity'], inplace = True)
fig, ax = plt.subplots(figsize = [15, 8])
sns.countplot(x = 'category', hue ='Clusters', data = exe)
ax.set(title = 'Comparison of Clusters Among the Categories',
       xlabel ='', ylabel = 'Count')
ax.set_xticklabels(ax.get_xticklabels(), rotation = 10)
plt.savefig('pca_clustering_plots/Comparison of Clusters Among the Categories')
```
