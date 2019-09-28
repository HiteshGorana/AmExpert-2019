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
data = pd.concat([train, test], sort=False).reset_index(drop = True)
ltr = len(train)
def get_importances(clfs):
    importances = [clf.feature_importance('gain') for clf in clfs]
    importances = np.vstack(importances)
    mean_gain = np.mean(importances, axis=0)
    features = clfs[0].feature_name()
    data = pd.DataFrame({'gain':mean_gain, 'feature':features})
    plt.figure(figsize=(8, 30))
    sns.barplot(x='gain', y='feature', data=data.sort_values('gain', ascending=False))
    plt.tight_layout()
    return data
def standart_split(data, n_splits):
    split_list = []
    for i in range(n_splits):
        kf = StratifiedKFold(n_splits=10, shuffle = True, random_state = 228)
        for train_index, test_index in kf.split(data.iloc[:ltr, :], data['redemption_status'][:ltr]):
            split_list += [(train_index, test_index)]
    return split_list

split_list = standart_split(data, 1)

def lgb_train(data, target, ltr, train_cols, split_list, param, n_e = 10000, cat_col = None, verb_num = None, imp=True):
    pred = pd.DataFrame()
    pred_val = np.zeros(ltr)
    score = []
    j = 0
    train_pred = pd.DataFrame()
    models = []
    for i , (train_index, test_index) in enumerate(split_list):
        param['seed'] = i
        tr = lgb.Dataset(np.array(data[train_cols])[train_index], np.array(data[target])[train_index])
        te = lgb.Dataset(np.array(data[train_cols])[test_index], np.array(data[target])[test_index], reference=tr)
        tt = lgb.Dataset(np.array(data[train_cols])[ltr:, :])
        evallist = [(tr, 'train'), (te, 'test')]
        bst = lgb.train(param, tr, num_boost_round = n_e,valid_sets = [tr, te], feature_name=train_cols,
                        early_stopping_rounds=150, verbose_eval = verb_num)
        pred[str(i)] =bst.predict(np.array(data[train_cols])[ltr:])
        pred_val[test_index] = bst.predict(np.array(data[train_cols])[test_index])
        score += [metrics.roc_auc_score(np.array(data[target])[test_index], pred_val[test_index])]
        models.append(bst)
        print(i, 'MEAN: ', np.mean(score), 'LAST: ', score[-1])
    if imp:
        get_importances(models)
        plt.show()
    train_pred[str(j)] = pred_val
    ans = pd.Series( pred.mean(axis = 1).tolist())
    ans.name = 'lgb'
    return pred, score, train_pred, bst

param_lgb = { 'boosting_type': 'gbdt', 'objective': 'binary', 'metric':'auc',
             'bagging_freq':1, 'subsample':1, 'feature_fraction': 0.7,
              'num_leaves': 8, 'learning_rate': 0.05, 'lambda_l1':5,'max_bin':255}

prediction, scores, oof, model = lgb_train(data, 'redemption_status', ltr, train_cols,
                       split_list, param_lgb,  verb_num  = 250)

tmp = prediction.copy()
for col in tmp.columns:
    tmp[col] = tmp[col].rank()
tmp = tmp.mean(axis = 1)
tmp  =tmp / tmp.max()
day = 1
sub = 3
name = f"day_{day}_sub_{sub}"
tmp = dict(zip(test.id.values, tmp))
answer1 = pd.DataFrame()
answer1['id'] = test.id.values
answer1['redemption_status'] = answer1['id'].map(tmp)
answer1.to_csv(f'{name}.csv', index = None)