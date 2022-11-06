```python
import matplotlib.pyplot as plt
import seaborn as sns
from plotting_tools import topcat
```

# In this phase we create some plots divided in three categories: 

- [Univariate](#Univariate)
- [Bivariate](#Bivariate)
- [Multivariate](#Multivariate)

## Univariate

### Top 10 Categories based on the Number of Apps
```python
plt.style.use("fivethirtyeight")
fig, ax = plt.subplots(figsize = [15, 8])
sns.countplot(x = 'category', data = topcat, palette = 'rainbow')
ax.set(title = 'Top 10 Categories based on the Number of Apps',
       xlabel ='Category', ylabel = 'Count')
ax.set_xticklabels(ax.get_xticklabels(), rotation = 10)
plt.savefig('eda/univariate/Top 10 Categories based on the Number of Apps.png')
```
### Distribution of Free Fature
```python
plt.figure(figsize = (15, 8))
plt.pie(topcat['free'].value_counts(), radius = 1, autopct = '%0.2f%%', explode = [0.1, 0.4], 
        labels = ['Free','Not Free'], startangle = 100, textprops = {"fontsize": 15})
plt.savefig('eda/univariate/Distribution of Free Feature.png')
``` 
### Distribution of Ad Supported Feature
```python
plt.figure(figsize = (15, 8))
plt.pie(topcat['ad_supported'].value_counts(), radius = 1, autopct = '%0.2f%%', explode = [0.1, 0.15], 
        labels = ['Ad Supported','Not Ad Supported'], startangle = 100, textprops = {"fontsize": 15});
plt.savefig('eda/univariate/Distribution of Ad Supported Feature.png')
```
### Distribution of In App Purchases Feature
```python
plt.figure(figsize = (15, 8))
plt.pie(topcat['in_app_purchases'].value_counts(), radius = 1, autopct = '%0.2f%%', explode = [0.1, 0.3], 
        labels = ['In App Purchases','Not In App Purchases'], startangle = 100, textprops = {"fontsize": 15});
plt.savefig('eda/univariate/Distribution of In App Purchases Feature.png')
```
### Content level of the Apps
```python
fig, ax = plt.subplots(figsize = [15, 8])
sns.countplot(x = 'content_rating', data = topcat, palette = 'rainbow')
ax.set(title = 'Content level of the Apps', 
       xlabel = 'Rating out of 5.0',
       ylabel = 'Count')
ax.set_xticklabels(ax.get_xticklabels(), rotation = 10)
plt.savefig('eda/univariate/Content level of the Apps.png')
```
### Rating of the Apps
```python
plt.figure(figsize = (15, 8))
plt.hist(topcat['rating'])
plt.xlabel('Rating')
plt.ylabel('Count')
plt.title('Rating of the Apps')
plt.savefig('eda/univariate/Rating of the Apps.png')
```
### Price of the Apps
```python
price_df = topcat[topcat["free"] == False]
plt.figure(figsize = (15, 8))
plt.hist(price_df['price'], 500)
plt.xlabel('Price')
plt.ylabel('Count')
plt.xlim(0, 35)
plt.title('Price of the Apps')
plt.savefig('eda/univariate/Price of the Apps.png')
```

## Bivariate

### Rating vs Installs
```python
plt.figure(figsize = [15, 8])
sns.lineplot(data = topcat, x = 'rating', y = 'installs')
plt.xlabel('Rating')
plt.ylabel('Installs')
plt.title('Rating vs Installs')
plt.savefig('eda/bivariate/Rating vs Installs.png')
```
### Comparison of Installs Among the Categories
```python
fig, ax = plt.subplots(figsize = [15, 8])
sns.barplot(x = 'category', y ='installs', data = topcat, palette = 'rainbow')
ax.set(title = 'Comparison of Installs Among the Categories',
       xlabel ='', ylabel = 'Installs')
ax.set_xticklabels(ax.get_xticklabels(), rotation = 10)
plt.savefig('eda/bivariate/Comparison of Installs Among the Categories.png')
```
### Comparison of Size Among the Categories
```python
fig, ax = plt.subplots(figsize = [15, 8])
sns.barplot(x = 'category', y ='size', data = topcat, palette = 'rainbow')
ax.set(title = 'Comparison of Size Among the Categories',
       xlabel ='', ylabel = 'Size (MB)')
ax.set_xticklabels(ax.get_xticklabels(), rotation = 10)
plt.savefig('eda/bivariate/Comparison of Size Among the Categories.png')
```
### Comparison of the Number of Ratings Among the Categories
```python
fig, ax = plt.subplots(figsize = [15, 8])
sns.barplot(x = 'category', y ='rating_count', data = topcat, palette = 'rainbow')
ax.set(title = 'Comparison of the Number of Ratings Among the Categories',
       xlabel ='', ylabel = 'Rating Count')
ax.set_xticklabels(ax.get_xticklabels(), rotation = 10)
plt.savefig('eda/bivariate/Comparison of the Number of Ratings Among the Categories.png')
```
### Rating vs Last Updated
```python
plt.figure(figsize = [15, 8])
sns.lineplot(data = topcat, x = 'last_updated', y = 'rating')
plt.xlabel('Last Updated')
plt.ylabel('Rating')
plt.title('Rating vs Last Updated')
plt.savefig('eda/bivariate/Rating vs Last Updated.png')
```

## Multivariate

### Correlation of the Numerical Features
```python
mask = np.zeros_like(topcat.corr())
mask[np.triu_indices_from(mask)] = True
plt.figure(figsize = [18, 12])
plt.title('Correlation of the Numerical Features')
a = sns.heatmap(topcat.corr(), center = 0, square = True, fmt = '.3f', annot = True, mask = mask)
a.set_xticklabels(a.get_xticklabels(), rotation = 10)
plt.savefig('eda/multivariate/Correlation of the Numerical Features.png')
```
### Comparison of Rating Between Categories & Ad Supported
```python
fig, ax = plt.subplots(figsize = [15, 8])
sns.barplot(x = 'category', y ='rating', data = topcat, palette = 'rainbow', hue  = 'ad_supported')
ax.set(title = 'Comparison of Rating Between Categories & Ad Supported',
       xlabel ='', ylabel = 'Rating')
ax.set_xticklabels(ax.get_xticklabels(), rotation = 10)
plt.savefig('eda/multivariate/Comparison of Rating Between Categories & Ad Supported.png')
```
### Comparison of Installs Between Categories & Free Apps
```python
fig, ax = plt.subplots(figsize = [15, 8])
sns.barplot(x = 'category', y ='installs', data = topcat, palette = 'rainbow', hue = 'free')
ax.set(title = 'Comparison of Installs Between Categories & Free Apps',
       xlabel ='', ylabel = 'Installs')
ax.set_xticklabels(ax.get_xticklabels(), rotation = 10)
plt.savefig('eda/multivariate/Comparison of Installs Between Categories & Free Apps.png')
```
