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
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures, RobustScaler
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score as auc
from sklearn.linear_model import LogisticRegression

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

data['gg_1_diff'] = data['campaign_id'].map(data.groupby('campaign_id')['customer_id'].apply(lambda x: len(x) - x.unique().size).to_dict())
data['gg_1_ratio'] = data['campaign_id'].map(data.groupby('campaign_id')['customer_id'].apply(lambda x: x.unique().size / len(x)).to_dict())

data['gg_2_diff'] = data['customer_id'].map(data.groupby('customer_id')['campaign_id'].apply(lambda x: len(x) - x.unique().size).to_dict())
data['gg_2_ratio'] = data['customer_id'].map(data.groupby('customer_id')['campaign_id'].apply(lambda x: x.unique().size / len(x)).to_dict())

data['gg_1_diff_'] = data['campaign_id'].map(data.groupby('campaign_id')['coupon_id'].apply(lambda x: len(x) - x.unique().size).to_dict())
data['gg_1_ratio_'] = data['campaign_id'].map(data.groupby('campaign_id')['coupon_id'].apply(lambda x: x.unique().size / len(x)).to_dict())

data['gg_2_diff_'] = data['coupon_id'].map(data.groupby('coupon_id')['campaign_id'].apply(lambda x: len(x) - x.unique().size).to_dict())
data['gg_2_ratio_'] = data['coupon_id'].map(data.groupby('coupon_id')['campaign_id'].apply(lambda x: x.unique().size / len(x)).to_dict())

data['diff_time_mean_campaign_id'] = data['campaign_id'].map(
    data.groupby('campaign_id')['start_date'].apply(lambda x: np.nanmean(x.diff() / np.timedelta64(1, 's'))).to_dict())
data['diff_time_max_campaign_id'] = data['campaign_id'].map(
    data.groupby('campaign_id')['start_date'].apply(lambda x: np.nanmax(x.diff() / np.timedelta64(1, 's'))).to_dict())
data['diff_time_min_campaign_id'] = data['campaign_id'].map(
    data.groupby('campaign_id')['start_date'].apply(lambda x: np.nanmin(x.diff() / np.timedelta64(1, 's'))).to_dict())

data['diff_time_mean_campaign_id_'] = data['campaign_id'].map(
    data.groupby('campaign_id')['end_date'].apply(lambda x: np.nanmean(x.diff() / np.timedelta64(1, 's'))).to_dict())
data['diff_time_max_campaign_id_'] = data['campaign_id'].map(
    data.groupby('campaign_id')['end_date'].apply(lambda x: np.nanmax(x.diff() / np.timedelta64(1, 's'))).to_dict())
data['diff_time_min_campaign_id_'] = data['campaign_id'].map(
    data.groupby('campaign_id')['end_date'].apply(lambda x: np.nanmin(x.diff() / np.timedelta64(1, 's'))).to_dict())

# data['diff_time_mean_coupon_id'] = data['coupon_id'].map(
#     data.groupby('coupon_id')['start_date'].apply(lambda x: np.nanmean(x.diff() / np.timedelta64(1, 's'))).to_dict())
# data['diff_time_max_coupon_id'] = data['coupon_id'].map(
#     data.groupby('coupon_id')['start_date'].apply(lambda x: np.nanmax(x.diff() / np.timedelta64(1, 's'))).to_dict())
# data['diff_time_min_coupon_id'] = data['coupon_id'].map(
#     data.groupby('coupon_id')['start_date'].apply(lambda x: np.nanmin(x.diff() / np.timedelta64(1, 's'))).to_dict())

# data['diff_time_mean_coupon_id_'] = data['coupon_id'].map(
#     data.groupby('coupon_id')['end_date'].apply(lambda x: np.nanmean(x.diff() / np.timedelta64(1, 's'))).to_dict())
# data['diff_time_max_coupon_id_'] = data['coupon_id'].map(
#     data.groupby('coupon_id')['end_date'].apply(lambda x: np.nanmax(x.diff() / np.timedelta64(1, 's'))).to_dict())
# data['diff_time_min_coupon_id_'] = data['coupon_id'].map(
#     data.groupby('coupon_id')['end_date'].apply(lambda x: np.nanmin(x.diff() / np.timedelta64(1, 's'))).to_dict())

# data['diff_time_mean_customer_id'] = data['customer_id'].map(
#     data.groupby('customer_id')['start_date'].apply(lambda x: np.nanmean(x.diff() / np.timedelta64(1, 's'))).to_dict())
# data['diff_time_max_customer_id'] = data['customer_id'].map(
#     data.groupby('customer_id')['start_date'].apply(lambda x: np.nanmax(x.diff() / np.timedelta64(1, 's'))).to_dict())
# data['diff_time_min_customer_id'] = data['customer_id'].map(
#     data.groupby('customer_id')['start_date'].apply(lambda x: np.nanmin(x.diff() / np.timedelta64(1, 's'))).to_dict())

# data['diff_time_mean_customer_id_'] = data['customer_id'].map(
#     data.groupby('customer_id')['end_date'].apply(lambda x: np.nanmean(x.diff() / np.timedelta64(1, 's'))).to_dict())
# data['diff_time_max_customer_id_'] = data['customer_id'].map(
#     data.groupby('customer_id')['end_date'].apply(lambda x: np.nanmax(x.diff() / np.timedelta64(1, 's'))).to_dict())
# data['diff_time_min_customer_id_'] = data['customer_id'].map(
#     data.groupby('customer_id')['end_date'].apply(lambda x: np.nanmin(x.diff() / np.timedelta64(1, 's'))).to_dict())


train_cols = [i for i in data.columns if i not in ['id','redemption_status','start_date','end_date']]
train = data[data['redemption_status'].notnull()]
test = data[data['redemption_status'].isnull()]
dummies = pd.get_dummies(data[train_cols].fillna(0), columns=train_cols, drop_first=True, sparse=True)
train_ohe = dummies.iloc[:train.shape[0], :]
test_ohe = dummies.iloc[train.shape[0]:, :]

print(train_ohe.shape)
print(test_ohe.shape)
train_ohe = train_ohe.sparse.to_coo().tocsr()
test_ohe = test_ohe.sparse.to_coo().tocsr()
# Model
def run_cv_model(train, test, target, model_fn, params={}, eval_fn=None, label='model'):
    kf = StratifiedKFold(n_splits=10, shuffle = True, random_state = 228)
    fold_splits = kf.split(train, target)
    cv_scores = []
    pred_full_test = 0
    pred_train = np.zeros((train.shape[0]))
    i = 1
    for dev_index, val_index in fold_splits:
        print('Started ' + label + ' fold ' + str(i) + '/5')
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
lr_params = {'solver': 'lbfgs','C': 1.8,'max_iter' : 2000}
results = run_cv_model(train_ohe, test_ohe, target, runLR, lr_params, auc, 'lr')
day = 3
sub = 3
name = f"day_{day}_sub_{sub}"
tmp = dict(zip(test.id.values, results['test']))
answer1 = pd.DataFrame()
answer1['id'] = test.id.values
answer1['redemption_status'] = answer1['id'].map(tmp)
answer1.to_csv(f'{name}.csv', index = None)