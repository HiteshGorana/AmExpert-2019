import pandas as pd
import numpy as np
import gc

import mlcrate as mlc
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
import pickle as pkl
import seaborn as sns
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential, Model
from keras.layers import Input, Embedding, Dense, Flatten, Concatenate, Dot, Reshape, Add, Subtract
from keras import objectives
from keras import backend as K
from keras import regularizers 
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.regularizers import l2
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from keras.layers import (Input, Lambda, Embedding, GaussianDropout, Reshape, CuDNNGRU,
                          BatchNormalization, Dropout, Dense, PReLU, Layer,ReLU, LeakyReLU,GRU, Bidirectional)
from keras.layers.merge import concatenate
from sklearn.model_selection import KFold, GroupKFold
from keras import callbacks
from keras.layers import (Input, Lambda, Embedding, GaussianDropout, Reshape, CuDNNGRU,
                          BatchNormalization, Dropout, Dense, PReLU, Layer,ReLU, LeakyReLU,GRU, Bidirectional)
def fallback_auc(y_true, y_pred):
    try:
        return roc_auc_score(y_true, y_pred)
    except:
        return 0.5

def auc(y_true, y_pred):
    return tf.py_function(fallback_auc, (y_true, y_pred), tf.double)

import numpy as np
import tensorflow as tf
import random as rn

# The below is necessary in Python 3.2.3 onwards to
# have reproducible behavior for certain hash-based operations.
# See these references for further details:
# https://docs.python.org/3.4/using/cmdline.html#envvar-PYTHONHASHSEED
# https://github.com/fchollet/keras/issues/2280#issuecomment-306959926

import os

def init_seeds(seed):
    os.environ['PYTHONHASHSEED'] = '0'

    # The below is necessary for starting Numpy generated random numbers
    # in a well-defined initial state.

    np.random.seed(seed)

    # The below is necessary for starting core Python generated random numbers
    # in a well-defined state.

    rn.seed(seed)

    # Force TensorFlow to use single thread.
    # Multiple threads are a potential source of
    # non-reproducible results.
    # For further details, see: https://stackoverflow.com/questions/42022950/which-seeds-have-to-be-set-where-to-realize-100-reproducibility-of-training-res

    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

    from keras import backend as K

    # The below tf.set_random_seed() will make random number generation
    # in the TensorFlow backend have a well-defined initial state.
    # For further details, see: https://www.tensorflow.org/api_docs/python/tf/set_random_seed

    tf.set_random_seed(seed)

    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)
    return sess
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
# day
customer_transaction_data['day'] = customer_transaction_data.date.dt.day
date_day_mean = customer_transaction_data.groupby("customer_id")['day'].mean().to_dict()
data['date_day_mean'] = data['customer_id'].map(date_day_mean)
#coupon_item_mapping, item_data
coupon_item_mapping = coupon_item_mapping.merge(item_data, how = 'left', on = 'item_id')
coupon_item_mapping['brand_type'] = pd.Series(coupon_item_mapping['brand_type'].factorize()[0]).replace(-1, np.nan)
coupon_item_mapping['category'] = pd.Series(coupon_item_mapping['category'].factorize()[0]).replace(-1, np.nan)

category = coupon_item_mapping.groupby("coupon_id")['category'].mean().to_dict()
data['category_mean'] = data['coupon_id'].map(category)
category = coupon_item_mapping.groupby("coupon_id")['category'].count().to_dict()
data['category_count'] = data['coupon_id'].map(category)
category = coupon_item_mapping.groupby("coupon_id")['category'].nunique().to_dict()
data['category_nunique'] = data['coupon_id'].map(category)
category = coupon_item_mapping.groupby("coupon_id")['category'].max().to_dict()
data['category_max'] = data['coupon_id'].map(category)
category = coupon_item_mapping.groupby("coupon_id")['category'].min().to_dict()
data['category_min'] = data['coupon_id'].map(category)

brand_mean = coupon_item_mapping.groupby("coupon_id")['brand'].mean().to_dict()
data['brand_mean'] = data['coupon_id'].map(brand_mean)
brand_mean = coupon_item_mapping.groupby("coupon_id")['brand'].count().to_dict()
data['brand_count'] = data['coupon_id'].map(brand_mean)
brand_mean = coupon_item_mapping.groupby("coupon_id")['brand'].min().to_dict()
data['brand_min'] = data['coupon_id'].map(brand_mean)
brand_mean = coupon_item_mapping.groupby("coupon_id")['brand'].max().to_dict()
data['brand_max'] = data['coupon_id'].map(brand_mean)
brand_mean = coupon_item_mapping.groupby("coupon_id")['brand'].nunique().to_dict()
data['brand_nunique'] = data['coupon_id'].map(brand_mean)

# selling_price
selling_price_mean = customer_transaction_data.groupby("customer_id")['selling_price'].mean().to_dict()
data['selling_price_mean'] = data['customer_id'].map(selling_price_mean)
selling_price_mean = customer_transaction_data.groupby("customer_id")['selling_price'].sum().to_dict()
data['selling_price_sum'] = data['customer_id'].map(selling_price_mean)
selling_price_mean = customer_transaction_data.groupby("customer_id")['selling_price'].min().to_dict()
data['selling_price_min'] = data['customer_id'].map(selling_price_mean)
selling_price_mean = customer_transaction_data.groupby("customer_id")['selling_price'].max().to_dict()
data['selling_price_max'] = data['customer_id'].map(selling_price_mean)
selling_price_mean = customer_transaction_data.groupby("customer_id")['selling_price'].nunique().to_dict()
data['selling_price_nunique'] = data['customer_id'].map(selling_price_mean)
train_cols = [i for i in data.columns if i not in ['id','redemption_status','start_date','end_date']]
train_cols = ['campaign_id','coupon_id','campaign_type','rented_mean','income_bracket_sum','age_range_mean','family_size_mean',
 'no_of_children_mean',
 'no_of_children_count',
 'marital_status_count',
 'quantity_mean',
 'coupon_discount_mean',
 'other_discount_mean',
 'date_day_mean',
 'category_mean',
 'category_nunique',
 'category_max',
 'category_min',
 'brand_mean',
 'brand_max',
 'brand_nunique',
 'selling_price_mean',
 'selling_price_min',
 'selling_price_nunique']
data[train_cols] = data[train_cols].fillna(0)
train = data[data['redemption_status'].notnull()]
test = data[data['redemption_status'].isnull()]

f_size  = [int(np.absolute(data[f]).max()) + 1 for f in train_cols]
k_latent = 2
embedding_reg = 0.0002
kernel_reg = 0.1

def get_embed(x_input, x_size, k_latent):
    if x_size > 0: #category
        embed = Embedding(x_size, k_latent, input_length=1, 
                          embeddings_regularizer=l2(embedding_reg))(x_input)
        embed = Flatten()(embed)
    else:
        embed = Dense(k_latent, kernel_regularizer=l2(embedding_reg))(x_input)
    return embed

def build_model_1(X, f_size):
    dim_input = len(f_size)
    
    input_x = [Input(shape=(1,)) for i in range(dim_input)] 
     
    biases = [get_embed(x, size, 2) for (x, size) in zip(input_x, f_size)]
    
    factors = [get_embed(x, size, k_latent) for (x, size) in zip(input_x, f_size)]
    
    s = Add()(factors)
    
    diffs = [Subtract()([s, x]) for x in factors]
    
    dots = [Dot(axes=1)([d, x]) for d,x in zip(diffs, factors)]
    
    x = Concatenate()(biases + dots)
    x = BatchNormalization()(x)
    output = Dense(1, activation='sigmoid', kernel_regularizer=l2(kernel_reg))(x)
    model = Model(inputs=input_x, outputs=[output])
    opt = Adam(clipnorm=0.2, lr=0.0031)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=[auc])
    output_f = factors + biases
    model_features = Model(inputs=input_x, outputs=output_f)
    return model, model_features
n_epochs = 100
P = 10
batch_size = 2**P
print(batch_size)

earlystopper = EarlyStopping(patience=0, verbose=1)
kf = StratifiedKFold(n_splits=10, shuffle = True, random_state = 228)
# kf = GroupKFold(5)

score = []
prediction = np.zeros(len(test))
validate = np.zeros(len(train))

test_ = [np.absolute(test[f].values) for f in train_cols]
y_train = train.redemption_status.values
w_train = (30 * (y_train > 0).astype('float32') + 1).ravel()

def schedule(epoch, lr):
    if epoch <= 10:
        lr = 0.0031
    if epoch > 10:
        lr = lr * 0.8
    return lr
lr_s = callbacks.LearningRateScheduler(schedule, verbose=1)
pred = pd.DataFrame()
for i , (tdx, vdx) in enumerate(kf.split(train, train.redemption_status)):
    try:
        del sess
    except:
        pass
    sess = init_seeds(i)
        
    print(f"FOLD : {i}")
    X_train = [np.absolute(train[f].iloc[tdx].values) for f in train_cols]
    X_test = [np.absolute(train[f].iloc[vdx].values) for f in train_cols]
    model, model_features = build_model_1(X_train, f_size)
    csv_logger = callbacks.CSVLogger(f'training_focal_loss{i}.log')
    model.fit(X_train,  y_train[tdx], 
          epochs=n_epochs, batch_size=batch_size, verbose=2, shuffle=True, 
          validation_data=(X_test,  y_train[vdx]), 
          callbacks=[earlystopper, csv_logger],

         )
    
    pred[str(i)] = model.predict(test_,verbose = False,batch_size=batch_size).reshape(-1)
    validate[vdx] = model.predict(X_test).reshape(-1)
    
    print(roc_auc_score(y_train[vdx], validate[vdx]))
    score.append(roc_auc_score(y_train[vdx], validate[vdx]))
    model.save_weights(f"model{i}.h5")
    del X_train, X_test,model, model_features
    gc.collect()
    
print(score)
print(f"CV : {np.mean(score)}")
    
    
tmp = pred.copy()
for col in tmp.columns:
    tmp[col] = tmp[col].rank()
    
tmp = tmp.mean(axis = 1)
tmp  =tmp / tmp.max()
day = 4
sub = 5
name = f"day_{day}_sub_{sub}"
tmp = dict(zip(test.id.values, tmp))
answer1 = pd.DataFrame()
answer1['id'] = test.id.values
answer1['redemption_status'] = answer1['id'].map(tmp)
answer1.to_csv(f'{name}.csv', index = None)