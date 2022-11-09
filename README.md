# **BT4222 Project: Stock Decision Prediction**

---
This project aims to predict and classify if one should buy, sell (i.e., short) or hold the S&P 500 stock index. With increasing volatility, this project investigates how we can use various sources of data to predict the stock index. With other studies typically focusing on only a single data source, our project utilizes diverse data sources, increasing the number of features to generate better stock predictions. 

We collected data from 1 January 2016 - 1 September 2022 (2,436 days), including:

* **S&P 500 Index Adjusted Close**: collected using `yfinance` package 
* **New York Times Articles**: collected using NYT API
* **Reddit Posts**: collected using Reddit API
* **FRED Economic Data**: downloaded and consolidated from the relevant CSV files


## Setting Up
1) Download data folder [here](https://drive.google.com/drive/folders/1A4eRk8IwxjMzR-nd1XVcDP9OEpo3ZBKa?usp=sharing)
2) Install all packages used: `pip install -r requirements.txt`

## Workflow
This project consists of 3 stages:
1. Data Generation
2. Data Preprocessing and Feature Engineering 
3. ML Models

### Data Generation
In this step, we collected New York Times article and Reddit Posts via their APIs. The codes can be found in the 'data_generation' folder.

(**NOTE**: These notebooks may take hours to run. Collected data can be found in "data/raw" folder.)
* `nyt_scrape.ipynb`: for collecting New York Times article
* `Get_Reddit.ipynb`: for collecting Reddit posts

### Data Preprocessing and Feature Engineering 
We conducted preprocessing on the data collected in the previous step and feature engineering to generate features for our ML models. We also conducted feature selection to select the best features and configuration. The codes can be found in the 'preprocessing_feature_engineering' folder. Steps are as follow:
1. Run `text_cleaning.ipynb` to preprocess collected textual data. This includes punctuation removal, emoji decoding to text, tokenization, stopwords removal and lemmatization with POS tagging.
2. Run `workflow.ipynb` to conduct feature engineering (e.g. sentiment analysis) and generate datasets of different configurations (i.e. percentage threshold and time lag). This notebook uses functions we defined in `feature_engineering.py`.
3. Run `logreg_base.ipynb` to determine the best configuration and features using a Logistic Regression model.
4. Run `eda.ipynb` to conduct exploratory data analysis on the dataset.

### ML Models
Different models using the best configuration and features were trained and tested. Hyperparameter tuning was also conducted. The codes can be found in the 'models' folder.
| Model                    | Notebook                   |
| ------------------------ | -------------------------- |
| Logistic Regression      | logreg_model.ipynb         |
| Decision tree            | decision_tree_model.ipynb  |
| Light Gradient Boosting  | lgmb_model.ipynb           |
| Random Forest            | random_forest_model.ipynb  |
| Linear SVM               | linear_svm_model.ipynb     |
| RBF SVM                  | rbf_svm_model.ipynb        |
| Multinomial Naive Bayes  | nb_model.ipynb             |
| Multilayer Pereceptron   | nn_model.ipynb             |


## Authors
* Chloe Wang Jia Qi - [chloewang10](https://github.com/chloewang10)
* Lam Pei Shi - [pspeishi](https://github.com/pspeishi)
* Yeo Hwee Kit, Sabrina - [yeonatalie](https://github.com/yeonatalie)
* Yeo Xiu Ying, Natalie - [sabyeo](https://github.com/sabyeo)
