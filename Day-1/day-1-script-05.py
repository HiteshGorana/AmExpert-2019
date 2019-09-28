import pandas as pd
import numpy as np
from datetime import timedelta
from tqdm import tqdm_notebook as tqdm
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
from matplotlib import pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
from scipy.stats import rankdata
from sklearn.preprocessing import StandardScaler, MinMaxScaler
file = {
	'test' : '../input/amexpert-2019/test.csv',
	'train':'../input/amexpert-2019/train.csv',
	'submission':'../input/amexpert-2019/submission.csv',
	'coupon_item_mapping' :'../input/amexpert-2019/coupon_item_mapping.csv',
	'campaign_data' : '../input/amexpert-2019/campaign_data.csv',
	'item_data' : '../input/amexpert-2019/item_data.csv',
	'customer_transaction_data':'../input/amexpert-2019/customer_transaction_data.csv',
	'customer_demographics':'../input/amexpert-2019/customer_demographics.csv',
}

train = pd.read_csv(file.get("train"))#
test = pd.read_csv(file.get("test"))#

coupon_item_mapping = pd.read_csv(file.get("coupon_item_mapping"))#No
item_data = pd.read_csv(file.get("item_data"))# may be yes
customer_transaction_data = pd.read_csv(file.get("customer_transaction_data"))#may be yes 

campaign_data = pd.read_csv(file.get("campaign_data"))#
customer_demographics = pd.read_csv(file.get("customer_demographics"))#
submission = pd.read_csv(file.get("submission"))
data = pd.concat([train, test], sort=False).reset_index(drop = True)
ltr = len(train)
data = data.merge(campaign_data, on='campaign_id')#  campaign_data
data = data.merge(customer_demographics, on='customer_id',how='left') #  customer_demographics
data = pd.merge_asof(data.sort_values("customer_id"), customer_transaction_data.sort_values("customer_id"), on='customer_id')
data = pd.merge_asof(data.sort_values("item_id"), item_data.sort_values("item_id"), on='item_id')
data['start_date'] = pd.to_datetime(data['start_date'])
data['end_date'] = pd.to_datetime(data['end_date'])
data['date'] = pd.to_datetime(data['date'])
data['difference'] = (data['end_date'] - data['start_date']) / np.timedelta64(1, 'D') # 1
data['end_date_month'] = data['end_date'].dt.month
data['end_date_dayofweek'] = data['end_date'].dt.dayofweek 
data['end_date_dayofyear'] = data['end_date'].dt.dayofyear 
data['end_date_days_in_month'] = data['end_date'].dt.days_in_month 
data['start_date_month'] = data['start_date'].dt.month
data['start_date_dayofweek'] = data['start_date'].dt.dayofweek 
data['start_date_dayofyear'] = data['start_date'].dt.dayofyear 
data['start_date_days_in_month'] = data['start_date'].dt.days_in_month
data['diff_dayofweek'] = data['end_date_dayofweek'] - data['start_date_dayofweek']
data['diff_dayofyear'] = data['end_date_dayofyear'] - data['start_date_dayofyear']

for i in data.columns:
    if str(data[i].dtype) == 'object':
        data[i] = data[i].factorize()[0]

train_cols = ['campaign_id', 'coupon_id', 'customer_id',
       'campaign_type', 'age_range','marital_status', 'rented', 'family_size', 'no_of_children',
       'income_bracket', 'item_id', 'quantity', 'selling_price',
       'other_discount', 'coupon_discount', 'brand', 'brand_type', 'category','difference', 'end_date_month', 'end_date_dayofweek',
       'end_date_dayofyear', 'end_date_days_in_month', 'start_date_month',
       'start_date_dayofweek', 'start_date_dayofyear',
       'start_date_days_in_month','diff_dayofweek','diff_dayofyear']

data[train_cols] = data[train_cols].fillna(data[train_cols].mean())
train = data[data['redemption_status'].notnull()]
test = data[data['redemption_status'].isnull()]

from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score as auc
from sklearn.linear_model import LogisticRegression

# Model
def run_cv_model(train, test, target, model_fn, params={}, eval_fn=None, label='model'):
    kf = StratifiedKFold(n_splits=10, shuffle = True, random_state = 228)
    fold_splits = kf.split(train, target)
    cv_scores = []
    pred_full_test = 0
    pred_train = np.zeros((train.shape[0]))
    i = 1
    for dev_index, val_index in fold_splits:
        print('Started ' + label + ' fold ' + str(i) + '/10')
        dev_X, val_X = train[dev_index], train[val_index]
        dev_y, val_y = target[dev_index], target[val_index]
        params2 = params.copy()
        pred_val_y, pred_test_y = model_fn(dev_X, dev_y, val_X, val_y, test, params2)
        pred_full_test = pred_full_test + pred_test_y
        pred_train[val_index] = pred_val_y
        if eval_fn is not None:
            cv_score = eval_fn(val_y, pred_val_y)
            cv_scores.append(cv_score)
            print(label + ' cv score {}: {}'.format(i, cv_score))
        i += 1
    print('{} cv scores : {}'.format(label, cv_scores))
    print('{} cv mean score : {}'.format(label, np.mean(cv_scores)))
    print('{} cv std score : {}'.format(label, np.std(cv_scores)))
    pred_full_test = pred_full_test / 10.0
    results = {'label': label,
              'train': pred_train, 'test': pred_full_test,
              'cv': cv_scores}
    return results


def runLR(train_X, train_y, test_X, test_y, test_X2, params):
    print('Train LR')
    model = LogisticRegression(**params)
    model.fit(train_X, train_y)
    print('Predict 1/2')
    pred_test_y = model.predict_proba(test_X)[:, 1]
    print('Predict 2/2')
    pred_test_y2 = model.predict_proba(test_X2)[:, 1]
    return pred_test_y, pred_test_y2
target = train['redemption_status'].values
sc = StandardScaler()
lr_params = {'solver': 'lbfgs','C': 2.8,'max_iter' : 3500}
results = run_cv_model(sc.fit_transform(train[train_cols].fillna(0).values), sc.fit_transform(test[train_cols].fillna(0).values), target, runLR, lr_params, auc, 'lr')
day = 1
sub = 5
name = f"day_{day}_sub_{sub}"
tmp = dict(zip(test.id.values, results['test']))
answer1 = pd.DataFrame()
answer1['id'] = test.id.values
answer1['redemption_status'] = answer1['id'].map(tmp)
answer1.to_csv(f'{name}.csv', index = None)