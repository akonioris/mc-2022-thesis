import pandas as pd
import time
import warnings
warnings.filterwarnings('ignore')

# This phase provides the following steps: 
# Loading the data
# Converting all columns into the correct type
# Dropping unnecessary features
# Dealing with missing data
# Removing duplicated rows
# Taking only the rows given by the Top 10 categories based on Number of Apps

def load_data(logger):
    start = time.time()
    try:
        df = pd.read_csv('Google-Playstore.csv')
        df.rename(lambda x: x.lower().strip().replace(' ', '_'), axis = 1, inplace = True)
        df['category'] = df['category'].str.replace('Music & Audio', 'Music')
        df['category'] = df['category'].str.replace('Educational', 'Education')
        df['size'] = pd.to_numeric(df['size'].str.replace(r'[k, M]', ''),
                                   errors = 'coerce', downcast = 'float')
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
    except Exception as e:
        return True, str(e)
        
    if topcat.shape[0] == 0:
        logger.error('Empty dataframe')
        return True, 'Empty Dataframe'
    
    end = time.time()
    logger.info(f"Execution time of load_data function is: {end - start} seconds")
    return topcat
    