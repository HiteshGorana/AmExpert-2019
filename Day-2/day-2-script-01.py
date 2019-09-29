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
#campaign_data, customer_demographics customer_transaction_data
# item_data, coupon_item_mapping
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
data['start_date'] = pd.to_datetime(data['start_date'], dayfirst=True)
data['end_date'] = pd.to_datetime(data['end_date'], dayfirst=True)
data['campaign_type'] = pd.Series(data['campaign_type'].factorize()[0]).replace(-1, np.nan)
#######################################################################
# customer_demographics
customer_demographics['no_of_children'] = customer_demographics['no_of_children'].replace('3+', 3).astype(float)
customer_demographics['family_size'] = customer_demographics['family_size'].replace('5+', 3).astype(float)
customer_demographics['marital_status'] = pd.Series(customer_demographics['marital_status'].factorize()[0]).replace(-1, np.nan)
customer_demographics['age_range'] = pd.Series(customer_demographics['age_range'].factorize()[0]).replace(-1, np.nan)

# rented
rented_mean = customer_demographics.groupby("customer_id")['rented'].mean().to_dict()
data['rented_mean'] = data['customer_id'].map(rented_mean)
# income_bracket
income_bracket_sum = customer_demographics.groupby("customer_id")['income_bracket'].sum().to_dict()
data['income_bracket_sum'] = data['customer_id'].map(income_bracket_sum)
# age_range
age_range_mean = customer_demographics.groupby("customer_id")['age_range'].mean().to_dict()
data['age_range_mean'] = data['customer_id'].map(age_range_mean)
# family_size
family_size_mean = customer_demographics.groupby("customer_id")['family_size'].mean().to_dict()
data['family_size_mean'] = data['customer_id'].map(family_size_mean)
# no_of_children
no_of_children_mean = customer_demographics.groupby("customer_id")['no_of_children'].mean().to_dict()
data['no_of_children_mean'] = data['customer_id'].map(no_of_children_mean)
no_of_children_count = customer_demographics.groupby("customer_id")['no_of_children'].count().to_dict()
data['no_of_children_count'] = data['customer_id'].map(no_of_children_count)
# marital_status
marital_status_count = customer_demographics.groupby("customer_id")['marital_status'].count().to_dict()
data['marital_status_count'] = data['customer_id'].map(marital_status_count)
#############################################################################
# customer_transaction_data
customer_transaction_data['date'] = pd.to_datetime(customer_transaction_data['date'])
# quantity	
quantity_mean = customer_transaction_data.groupby("customer_id")['quantity'].mean().to_dict()
data['quantity_mean'] = data['customer_id'].map(quantity_mean)
#coupon_discount
coupon_discount_mean = customer_transaction_data.groupby("customer_id")['coupon_discount'].mean().to_dict()
data['coupon_discount_mean'] = data['customer_id'].map(coupon_discount_mean)
# other_discount
other_discount_mean = customer_transaction_data.groupby("customer_id")['other_discount'].mean().to_dict()
data['other_discount_mean'] = data['customer_id'].map(other_discount_mean)
# selling_price
selling_price_mean = customer_transaction_data.groupby("customer_id")['selling_price'].mean().to_dict()
data['selling_price_mean'] = data['customer_id'].map(selling_price_mean)
# day
customer_transaction_data['day'] = customer_transaction_data.date.dt.day
date_day_mean = customer_transaction_data.groupby("customer_id")['day'].mean().to_dict()
data['date_day_mean'] = data['customer_id'].map(date_day_mean)
#coupon_item_mapping, item_data
coupon_item_mapping = coupon_item_mapping.merge(item_data, how = 'left', on = 'item_id')
coupon_item_mapping['brand_type'] = pd.Series(coupon_item_mapping['brand_type'].factorize()[0]).replace(-1, np.nan)
coupon_item_mapping['category'] = pd.Series(coupon_item_mapping['category'].factorize()[0]).replace(-1, np.nan)

brand_mean = coupon_item_mapping.groupby("coupon_id")['brand'].mean().to_dict()
data['brand_mean'] = data['coupon_id'].map(brand_mean)

brand_type_mean = coupon_item_mapping.groupby("coupon_id")['brand_type'].mean().to_dict()
data['brand_type_mean'] = data['coupon_id'].map(brand_type_mean)

category_mean = coupon_item_mapping.groupby("coupon_id")['category'].mean().to_dict()
data['category_mean'] = data['coupon_id'].map(category_mean)

value_list_diff_mean = []
value_list_diff_max = []
value_list_diff_min = []
for customer_id in data.customer_id.values:
    cur_data = customer_transaction_data[customer_transaction_data.customer_id == customer_id]
    value_list_diff_mean += [cur_data['date'].diff().mean()]
    value_list_diff_max += [cur_data['date'].diff().max()]
    value_list_diff_min += [cur_data['date'].diff().min()]
data['diff_mean'] = value_list_diff_mean 
data['diff_max'] = value_list_diff_max
data['diff_min'] = value_list_diff_min
data['diff_mean'] = data['diff_mean'] / np.timedelta64(1, 's')
data['diff_max'] = data['diff_max'] / np.timedelta64(1, 's')
data['diff_min'] = data['diff_min'] / np.timedelta64(1, 's')

value_list_diff_mean_ = []
value_list_diff_max_ = []
value_list_diff_min_ = []
value_list_diff_mean__ = []
value_list_diff_max__ = []
value_list_diff_min__ = []
for customer_id in data.customer_id.values:
    cur_data = data[data.customer_id == customer_id]
    value_list_diff_mean_ += [cur_data['start_date'].diff().mean()]
    value_list_diff_max_ += [cur_data['start_date'].diff().max()]
    value_list_diff_min_ += [cur_data['start_date'].diff().min()]
    value_list_diff_mean__ += [cur_data['end_date'].diff().mean()]
    value_list_diff_max__ += [cur_data['end_date'].diff().max()]
    value_list_diff_min__ += [cur_data['end_date'].diff().min()]
data['diff_mean_'] = value_list_diff_mean_ 
data['diff_max_'] = value_list_diff_max_
data['diff_min_'] = value_list_diff_min_
data['diff_mean_'] = data['diff_mean_'] / np.timedelta64(1, 's')
data['diff_max_'] = data['diff_max_'] / np.timedelta64(1, 's')
data['diff_min_'] = data['diff_min_'] / np.timedelta64(1, 's')
data['diff_mean__'] = value_list_diff_mean__ 
data['diff_max__'] = value_list_diff_max__
data['diff_min__'] = value_list_diff_min__
data['diff_mean__'] = data['diff_mean__'] / np.timedelta64(1, 's')
data['diff_max__'] = data['diff_max__'] / np.timedelta64(1, 's')
data['diff_min__'] = data['diff_min__'] / np.timedelta64(1, 's')

train_cols = [i for i in data.columns if i not in ['id','redemption_status','start_date','end_date']]

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

def lgb_train(data, target, ltr, train_cols, split_list, param, n_e = 10000, cat_col = None, verb_num = None, imp=False):
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
day = 2
sub = 1
name = f"day_{day}_sub_{sub}"
tmp = dict(zip(test.id.values, tmp))
answer1 = pd.DataFrame()
answer1['id'] = test.id.values
answer1['redemption_status'] = answer1['id'].map(tmp)
answer1.to_csv(f'{name}.csv', index = None)