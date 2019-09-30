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
train = pd.read_csv(file.get("train"))#
test = pd.read_csv(file.get("test"))#
coupon_item_mapping = pd.read_csv(file.get("coupon_item_mapping"))#No
item_data = pd.read_csv(file.get("item_data"))# may be yes
customer_transaction_data = pd.read_csv(file.get("customer_transaction_data"))#may be yes 
campaign_data = pd.read_csv(file.get("campaign_data"))#
customer_demographics = pd.read_csv(file.get("customer_demographics"))#
data = pd.concat([train, test], sort=False).reset_index(drop = True)
ltr = len(train)
data = data.join(customer_demographics, on='customer_id',lsuffix='_left', rsuffix='_right')
data = data.join(campaign_data, on='campaign_id',lsuffix='_left', rsuffix='_right')

A = coupon_item_mapping.join(item_data, how='inner',lsuffix='_left', rsuffix='_right')
B = customer_transaction_data.join(item_data, how='inner',lsuffix='_left', rsuffix='_right')

A['brand_type'] = A['brand_type'].factorize()[0]
A['category'] = A['category'].factorize()[0]

brand_rename = {}
for i in ['min','max','count','mean','median']:
    brand_rename[i] = f'brand_{i}'
brand = A.groupby('coupon_id')['brand'].agg(['min','max','count','mean','median']).rename(columns = brand_rename)
brand['brand_unique'] = A.groupby('coupon_id')['brand'].nunique()

brand_type_rename = {}
for i in ['min','max','count','mean','median']:
    brand_type_rename[i] = f'brand_type_{i}'
brand_type = A.groupby('coupon_id')['brand_type'].agg(['min','max','count','mean','median']).rename(columns = brand_type_rename)
brand_type['brand_type_unique'] = A.groupby('coupon_id')['brand_type'].nunique()

category_rename = {}
for i in ['min','max','count','mean','median']:
    category_rename[i] = f'category_{i}'
category = A.groupby('coupon_id')['category'].agg(['min','max','count','mean','median']).rename(columns = category_rename)
category['category_unique'] = A.groupby('coupon_id')['category'].nunique()

data = pd.merge(data,brand,on=['coupon_id'],how='left')
data = pd.merge(data,brand_type,on=['coupon_id'],how='left')
data = pd.merge(data,category,on=['coupon_id'],how='left')

B['brand_type'] = B['brand_type'].factorize()[0]
B['category'] = B['category'].factorize()[0]

quantity_rename = {}
for i in ['min','max','count','mean','median']:
    quantity_rename[i] = f'quantity_{i}'
quantity = B.groupby('customer_id')['quantity'].agg(['min','max','count','mean','median']).rename(columns = quantity_rename)
quantity['quantity_unique'] = B.groupby('customer_id')['quantity'].nunique()

selling_price_rename = {}
for i in ['min','max','count','mean','median']:
    selling_price_rename[i] = f'selling_price_{i}'
selling_price = B.groupby('customer_id')['selling_price'].agg(['min','max','count','mean','median']).rename(columns = selling_price_rename)
selling_price['selling_price_unique'] = B.groupby('customer_id')['selling_price'].nunique()

other_discount_rename = {}
for i in ['min','max','count','mean','median']:
    other_discount_rename[i] = f'other_discount_{i}'
other_discount = B.groupby('customer_id')['other_discount'].agg(['min','max','count','mean','median']).rename(columns = other_discount_rename)
other_discount['other_discount_unique'] = B.groupby('customer_id')['other_discount'].nunique()

coupon_discount_rename = {}
for i in ['min','max','count','mean','median']:
    coupon_discount_rename[i] = f'coupon_discount_{i}'
coupon_discount = B.groupby('customer_id')['coupon_discount'].agg(['min','max','count','mean','median']).rename(columns = coupon_discount_rename)
coupon_discount['coupon_discount_unique'] = B.groupby('customer_id')['coupon_discount'].nunique()

brand_rename = {}
for i in ['min','max','count','mean','median']:
    brand_rename[i] = f'brand__{i}'
brand = B.groupby('customer_id')['brand'].agg(['min','max','count','mean','median']).rename(columns = brand_rename)
brand['brand__unique'] = B.groupby('customer_id')['brand'].nunique()

brand_type_rename = {}
for i in ['min','max','count','mean','median']:
    brand_type_rename[i] = f'brand_type__{i}'
brand_type = B.groupby('customer_id')['brand_type'].agg(['min','max','count','mean','median']).rename(columns = brand_type_rename)
brand_type['brand_type__unique'] = B.groupby('customer_id')['brand_type'].nunique()

category_rename = {}
for i in ['min','max','count','mean','median']:
    category_rename[i] = f'category__{i}'
category = B.groupby('customer_id')['category'].agg(['min','max','count','mean','median']).rename(columns = category_rename)
category['category__unique'] = B.groupby('customer_id')['category'].nunique()
data = data.rename(columns = {"customer_id_left" :'customer_id'})
del data['customer_id_right']
del data['campaign_id_right']
data = pd.merge(data,quantity,on=['customer_id'],how='left')
data = pd.merge(data,selling_price,on=['customer_id'],how='left')
data = pd.merge(data,other_discount,on=['customer_id'],how='left')
data = pd.merge(data,coupon_discount,on=['customer_id'],how='left')
data = pd.merge(data,brand,on=['customer_id'],how='left')
data = pd.merge(data,brand_type,on=['customer_id'],how='left')
data = pd.merge(data,category,on=['customer_id'],how='left')
for i in data.select_dtypes('object').columns.to_list():
    data[i] = pd.Series(data[i].factorize()[0]).replace(-1, np.nan)
    
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
    pred_full_test = pd.Series(pred_full_test).rank() / 10.0
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
day = 2
sub = 3
name = f"day_{day}_sub_{sub}"
tmp = dict(zip(test.id.values, results['test']))
answer1 = pd.DataFrame()
answer1['id'] = test.id.values
answer1['redemption_status'] = answer1['id'].map(tmp)
answer1.to_csv(f'{name}.csv', index = None)