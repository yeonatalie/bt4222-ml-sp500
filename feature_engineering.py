import pandas as pd
import numpy as np
import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import ast

def get_growth(df):
    '''
    Helper function for macro_feature_engineer method
    '''
    new_df = pd.DataFrame()
    for col_name in df.columns:
        new_col_name = col_name.replace("Actual", "Growth")
        actual = df[[col_name]].dropna().rename(columns={col_name: new_col_name})
        actual_lag = actual.shift(1)
        growth = (actual - actual_lag) / actual_lag
        growth = growth.reindex(df.index)
        new_df[new_col_name] = growth[new_col_name]
    return new_df


def macro_imputation(df):
    '''
    Helper function for macro_feature_engineer method
    '''
    df = df[df['Date'] <= '2022-09'].set_index('Date')
    df.index = pd.date_range(start='2016-01-01', end='2022-09-01', freq="MS")
    df = df.reindex(pd.date_range(start='2016-01-01', end='2022-09-01', freq="D")).fillna(method='ffill')
    return df

def macro_feature_engineer(df, normalize=False, scaler_type="standard", data_type="actual"):
    '''
    Feature engineering for macroeconomic data

    Input
    =====
    df: pd.DataFrame; contains actual macroeconomic data from 2016 to 2018
    normalize: bool; conducts normalization if set to True
    scaler_type: {"standard", "minmax"}; type of scaler for normalization
    data_type: {"actual", "growth", "both"}; type of features returned

    Output
    ======
    pd.DataFrame after feature engineering
    '''
    if normalize:
        scaler = StandardScaler() if scaler_type=="standard" else MinMaxScaler()
        train_scaled = scaler.fit_transform(df.loc[:'2022-01-01'])
        test_scaled = scaler.transform(df.loc['2022-01-01':])
        df_scaled = pd.DataFrame(np.vstack([train_scaled, test_scaled]))
        df_scaled.columns = df.columns
        df_scaled.index = df.index
        df = df_scaled

    if data_type == "actual":
        return macro_imputation(df)
    elif data_type == "growth":
        return macro_imputation(get_growth(df))
    else:
        growth_df = get_growth(df)
        return macro_imputation(pd.concat([df, growth_df], axis=1))


def reddit_feature_engineer(df):
    '''
    Feature engineering for reddit posts

    Input
    =====
    df: pd.DataFrame; contains reddit post datetime, cleaned_text, num_comments, score

    Output
    ======
    pd.DataFrame containing daily aggregated sentiment scores (3 types of aggregation)
    '''
    pos, neg, neu, compound = [], [], [], []
    analyzer = SentimentIntensityAnalyzer()
    for text in df['cleaned_text']:
        sentiment_dict = analyzer.polarity_scores(text)
        pos.append(sentiment_dict['pos'])
        neg.append(sentiment_dict['neg'])
        neu.append(sentiment_dict['neu'])
        compound.append(sentiment_dict['compound'])

    new_df = df[['author', 'id', 'num_comments', 'score', 'subreddit', 'cleaned_text']]
    new_df['date'] = [x.date() for x in df['datetime']]
    new_df['pos_score'] = pos
    new_df['neg_score'] = neg
    new_df['neu_score'] = neu
    new_df['compound_score'] = compound

    total_daily_comments_dict = dict(new_df.groupby('date').sum()['num_comments'])
    total_daily_upvotes_dict = dict(new_df.groupby('date').sum()['score'])
    total_daily_posts_dict = dict(new_df.groupby('date').count()['cleaned_text'])

    pos_score_weighted_comments, neg_score_weighted_comments, neu_score_weighted_comments, compound_score_weighted_comments = [], [], [], []
    pos_score_weighted_upvotes, neg_score_weighted_upvotes, neu_score_weighted_upvotes, compound_score_weighted_upvotes = [], [], [], []
    pos_score_weighted_both, neg_score_weighted_both, neu_score_weighted_both, compound_score_weighted_both = [], [], [], []

    for i in range(len(new_df)):
        # weighted by comments
        weight_comments = (new_df.iloc[i]['num_comments']+1)/(total_daily_comments_dict[new_df.iloc[i]['date']]+total_daily_posts_dict[new_df.iloc[i]['date']])
        pos_score_weighted_comments.append(weight_comments * new_df.iloc[i]['pos_score'])
        neg_score_weighted_comments.append(weight_comments * new_df.iloc[i]['neg_score'])
        neu_score_weighted_comments.append(weight_comments * new_df.iloc[i]['neu_score'])
        compound_score_weighted_comments.append(weight_comments * new_df.iloc[i]['compound_score'])

        # weighted by upvotes
        weight_upvotes = (new_df.iloc[i]['score']+1)/(total_daily_upvotes_dict[new_df.iloc[i]['date']]+total_daily_posts_dict[new_df.iloc[i]['date']])
        pos_score_weighted_upvotes.append(weight_upvotes * new_df.iloc[i]['pos_score'])
        neg_score_weighted_upvotes.append(weight_upvotes * new_df.iloc[i]['neg_score'])
        neu_score_weighted_upvotes.append(weight_upvotes * new_df.iloc[i]['neu_score'])
        compound_score_weighted_upvotes.append(weight_upvotes * new_df.iloc[i]['compound_score'])

        # weighted by both comments and upvotes
        weight_both = 0.5*((new_df.iloc[i]['num_comments']+1)/(total_daily_comments_dict[new_df.iloc[i]['date']]+total_daily_posts_dict[new_df.iloc[i]['date']])) + \
                    0.5*((new_df.iloc[i]['score']+1)/(total_daily_upvotes_dict[new_df.iloc[i]['date']]+total_daily_posts_dict[new_df.iloc[i]['date']]))
        pos_score_weighted_both.append(weight_both * new_df.iloc[i]['pos_score'])
        neg_score_weighted_both.append(weight_both * new_df.iloc[i]['neg_score'])
        neu_score_weighted_both.append(weight_both * new_df.iloc[i]['neu_score'])
        compound_score_weighted_both.append(weight_both * new_df.iloc[i]['compound_score'])

    new_df['pos_score_weighted_comments'] = pos_score_weighted_comments
    new_df['neg_score_weighted_comments'] = neg_score_weighted_comments
    new_df['neu_score_weighted_comments'] = neu_score_weighted_comments
    new_df['compound_score_weighted_comments'] = compound_score_weighted_comments

    new_df['pos_score_weighted_upvotes'] = pos_score_weighted_upvotes
    new_df['neg_score_weighted_upvotes'] = neg_score_weighted_upvotes
    new_df['neu_score_weighted_upvotes'] = neu_score_weighted_upvotes
    new_df['compound_score_weighted_upvotes'] = compound_score_weighted_upvotes

    new_df['pos_score_weighted_both'] = pos_score_weighted_both
    new_df['neg_score_weighted_both'] = neg_score_weighted_both
    new_df['neu_score_weighted_both'] = neu_score_weighted_both
    new_df['compound_score_weighted_both'] = compound_score_weighted_both

    # get aggregated scores for each day by summing the scores
    reddit_daily_df = new_df.groupby('date').sum()[['pos_score', 'neg_score', 'neu_score', 'compound_score', 'pos_score_weighted_comments', 'neg_score_weighted_comments', 'neu_score_weighted_comments', 'compound_score_weighted_comments', 'pos_score_weighted_upvotes', 'neg_score_weighted_upvotes', 'neu_score_weighted_upvotes', 'compound_score_weighted_upvotes', 'pos_score_weighted_both', 'neg_score_weighted_both', 'neu_score_weighted_both', 'compound_score_weighted_both']].sort_index()

    # reindex to fill rows with NA values if no post on that day
    reddit_daily_df = reddit_daily_df.reindex(pd.date_range(start='2016-01-01', end='2022-09-01'))
    reddit_daily_df = reddit_daily_df.reset_index().rename(columns={'index':'date'})

    # impute NA values using mean of last 3 days
    null_idx = reddit_daily_df[reddit_daily_df.isnull().any(axis=1)].index
    score_cols = ['pos_score', 'neg_score', 'neu_score', 'compound_score',
                    'pos_score_weighted_comments', 'neg_score_weighted_comments',
                    'neu_score_weighted_comments', 'compound_score_weighted_comments',
                    'pos_score_weighted_upvotes', 'neg_score_weighted_upvotes',
                    'neu_score_weighted_upvotes', 'compound_score_weighted_upvotes',
                    'pos_score_weighted_both', 'neg_score_weighted_both',
                    'neu_score_weighted_both', 'compound_score_weighted_both']
    for i in null_idx:
        for col in score_cols:
            reddit_daily_df.at[i,col] = (reddit_daily_df.at[i-1,col]+reddit_daily_df.at[i-2,col]+reddit_daily_df.at[i-3,col])/3

    return reddit_daily_df.set_index('date')

def nyt_feature_engineer(df, df_weights):
    df_end = len(df)

    df_out = df[['date', 'keyword', 'headline', 'abstract', 'lead_paragraph', 'section', 'hits', 'word_count', 'cleaned_text']]
    pos, neg, neu = [],  [], [] 


    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")


    for i in range(0,df_end): 
        df_array = np.array(df.iloc[i]['cleaned_text'])
        df_list = df_array.tolist()
        inputs = tokenizer(df_list, padding = True, truncation = True, return_tensors='pt')
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        positive = predictions[:, 0].tolist()
        pos.append(positive[0])
        
        negative = predictions[:, 1].tolist()
        neg.append(negative[0])
        
        neutral = predictions[:, 2].tolist()
        neu.append(neutral[0])
    df_out['pos_score'] = pos
    df_out['neg_score'] = neg
    df_out['neu_score'] = neu


    clean_keyword = [] 


    for (index, row) in df_out.iterrows(): 
        changed = False
        row['keyword'] = ast.literal_eval(row['keyword'])
        keyword = row['keyword']
        if 'Alphabet' in row['keyword']:
            changed = True
            keyword.extend(list(set(map(lambda x: x.replace('Alphabet', 'Google'), row['keyword']))))
        if 'GOOG' in row['keyword']: 
            changed = True
            keyword.extend(list(set(map(lambda x: x.replace('GOOG', 'Google'), row['keyword']))))
        if 'Berkshire Hathaway' in row['keyword']: 
            changed = True
            keyword.extend(list(set(map(lambda x: x.replace('Berkshire Hathaway', 'BRK'), row['keyword']))))
        if 'Johnson' in row['keyword']:
            changed = True
            keyword.extend(list(set(map(lambda x: x.replace('Johnson', 'JnJ'), row['keyword']))))
        if 'ExxonMobil' in row['keyword']: 
            changed = True
            keyword.extend(list(set(map(lambda x: x.replace('ExxonMobil', 'Exxon'), row['keyword']))))
        if 'Facebook' in row['keyword']: 
            changed = True
            keyword.extend(list(set(map(lambda x: x.replace('Facebook', 'Meta'), row['keyword']))))
        if 'JP Morgan' in row['keyword']: 
            changed = True
            keyword.extend(list(set(map(lambda x: x.replace('JP Morgan', 'JPMorgan'), row['keyword']))))
        if 'Thermo Fisher Scientific' in row['keyword']: 
            changed = True
            keyword.extend(list(set(map(lambda x: x.replace('Thermo Fisher Scientific', 'Thermo Fisher'), row['keyword']))))
        if 'Pepsi' in row['keyword']: 
            changed = True
            keyword.extend(list(set(map(lambda x: x.replace('Pepsi', 'PepsiCo'), row['keyword']))))
        if 'S. P.' in row['keyword']: 
            changed = True
            keyword.extend(list(set(map(lambda x: x.replace('S. P.', 'SP500'), row['keyword']))))
        if 'Wall Street' in row['keyword']: 
            changed = True
            keyword.extend(list(set(map(lambda x: x.replace('Wall Street', 'SP500'), row['keyword']))))
        if 'Fed' in row['keyword']: 
            changed = True
            keyword.extend(list(set(map(lambda x: x.replace('Fed', 'SP500'), row['keyword']))))
        if 'Federal Reserve' in row['keyword']: 
            changed = True
            keyword.extend(list(set(map(lambda x: x.replace('Federal Reserve', 'SP500'), row['keyword']))))
        if changed == False: 
            clean_keyword.append(row['keyword'])
        else: 
            clean_keyword.append(keyword)


    df_out['clean_keyword'] = clean_keyword

    ## weighting
    df_weights = df_weights.drop(['#'], axis = 1)
    df_weights = df_weights.set_index('Company').T.to_dict('list')

    weights = []
    for (index, row) in df_out.iterrows(): 
        weight = 0 
        changed = False 
        if 'Apple' in row['clean_keyword']: 
            changed = True
            weight += df_weights['Apple'][0]
            
        if 'Microsoft' in row['clean_keyword']: 
            changed = True
            weight += df_weights['Microsoft'][0]
        
        if 'Amazon' in row['clean_keyword']: 
            changed = True
            weight += df_weights['Amazon'][0]
        
        if 'Google' in row['clean_keyword']:  
            changed = True
            weight += df_weights['Google'][0]
        
        if 'Tesla' in row['clean_keyword']:   
            changed = True
            weight += df_weights['Tesla'][0]
        
        if 'BRK' in row['clean_keyword']:    
            changed = True
            weight += df_weights['BRK'][0]
        
        if 'UnitedHealth' in row['clean_keyword']:  
            changed = True
            weight += df_weights['UnitedHealth'][0]
        
        if 'JnJ' in row['clean_keyword']:  
            changed = True
            weight += df_weights['JnJ'][0]
        
        if 'Exxon' in row['clean_keyword']: 
            changed = True
            weight += df_weights['Exxon'][0]
        
        if 'JPMorgan' in row['clean_keyword']: 
            changed = True
            weight += df_weights['JPMorgan'][0]
        
        if 'Procter' in row['clean_keyword']:
            changed = True
            weight += df_weights['Procter'][0]
        
        if 'Visa' in row['clean_keyword']: 
            changed = True
            weight += df_weights['Visa'][0]
        
        if 'NVIDIA' in row['clean_keyword']: 
            changed = True
            weight += df_weights['NVIDIA'][0]
        
        if 'Chevron Corporation' in row['clean_keyword']: 
            changed = True
            weight += df_weights['Chevron Corporation'][0]
        
        if 'Meta' in row['clean_keyword']: 
            changed = True
            weight += df_weights['Meta'][0]
        
        if 'Home Depot' in row['clean_keyword']: 
            changed = True
            weight += df_weights['Home Depot'][0]

        if 'Eli Lilly and Company' in row['clean_keyword']: 
            changed = True
            weight += df_weights['Eli Lilly and Company'][0]

        if 'AbbVie' in row['clean_keyword']: 
            changed = True
            weight += df_weights['AbbVie'][0]

        if 'Mastercard' in row['clean_keyword']: 
            changed = True
            weight += df_weights['Mastercard'][0]

        if 'Pfizer' in row['clean_keyword']: 
            changed = True
            weight += df_weights['Pfizer'][0]
            
        if 'PepsiCo' in row['clean_keyword']: 
            changed = True
            weight += df_weights['PepsiCo'][0]

        if 'Merck' in row['clean_keyword']: 
            changed = True
            weight += df_weights['Merck'][0]
            
        if 'Bank of America' in row['clean_keyword']: 
            changed = True
            weight += df_weights['Bank of America'][0]
        if 'Coca-Cola' in row['clean_keyword']: 
            changed = True
            weight += df_weights['Coca-Cola'][0]

        if 'Costco' in row['clean_keyword']: 
            changed = True
            weight += df_weights['Costco'][0]

        if 'Thermo Fisher' in row['clean_keyword']: 
            changed = True
            weight += df_weights['Thermo Fisher'][0]
            
        if 'Walmart' in row['clean_keyword']: 
            changed = True
            weight += df_weights['Walmart'][0]

        if 'McDonald\'s' in row['clean_keyword']: 
            changed = True
            weight += df_weights['McDonald\'s'][0]

        if 'Broadcom' in row['clean_keyword']: 
            changed = True
            weight += df_weights['Broadcom'][0]

        if 'Walt Disney' in row['clean_keyword']: 
            changed = True
            weight += df_weights['Walt Disney'][0]

        if 'SP500' in row['clean_keyword']: 
            changed = True
            weight += df_weights['SP500'][0]

        if changed == False: 
            weight += 0
            changed = True
            
        else: 
            weights.append(weight)

    df_out['weights'] = weights

    pos_weight, neg_weight, neu_weight = [], [], []

    for i in range(len(df_out)): 
        pos_weight.append(df_out.iloc[i]['weights'] / 100 * df_out.iloc[i]['pos_score'])
        neg_weight.append(df_out.iloc[i]['weights'] / 100 * df_out.iloc[i]['neg_score'])
        neu_weight.append(df_out.iloc[i]['weights'] / 100 * df_out.iloc[i]['neu_score'])

    df_out['pos_weighted'] = pos_weight
    df_out['neg_weighted'] = neg_weight
    df_out['neu_weighted'] = neu_weight

    nyt_daily = df_out.groupby('date').mean()[['pos_score', 'neg_score', 'neu_score', 'pos_weighted', 'neg_weighted', 'neu_weighted']].sort_index()

    nyt_daily = nyt_daily.set_index('date')
    nyt_scores = nyt_scores.reindex(pd.date_range(start="2016-01-01", end="2022-09-01"))
    nyt_scores['date'] = nyt_scores.index
    nyt_scores = nyt_scores.reset_index(drop = True)

    # impute NA values using mean of last 3 days
    null_idx = nyt_daily[nyt_daily.isnull().any(axis=1)].index
    score_cols = ['pos_score', 'neg_score', 'neu_score', 'pos_weighted','neg_weighted','neu_weighted']
    for i in null_idx:
        for col in score_cols:
            nyt_daily.at[i,col] = (nyt_daily.at[i-1,col]+nyt_daily.at[i-2,col]+nyt_daily.at[i-3,col])/3
            
    return nyt_daily
        

def compute_lagged_values(df, days=3, method="mean"):
    '''
    Compute lagged values

    Input
    =====
    df: pd.DataFrame; contains Date as index and daily feature values
    days: int; number of daily lags
    method: {"mean", "median"}; aggregation method

    Output
    ======
    pd.DataFrame containing aggregated lagged values
    '''
    final_df = pd.DataFrame()
    for col in df.columns:
        lag_dfs = []
        for i in range(1, days+1):
            df_lag = df.shift(i)
            lag_dfs.append(df_lag[col])
        temp = pd.concat(lag_dfs)
        if method == "mean":
            final_df[col] = temp.groupby(temp.index).mean()
        elif method == "median":
            final_df[col] = temp.groupby(temp.index).median()
        else:
            raise Exception('method takes values {"mean", "median"}')
    return final_df