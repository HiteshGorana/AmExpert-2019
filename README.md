# [AmExpert-2019-Machine-Learning-Hackathon](https://datahack.analyticsvidhya.com/contest/amexpert-2019-machine-learning-hackathon)
![](./img.png)
 Machine Learning Hackathon
 
  ```python
 #  Feature Engenering
data['campaign_id_exp_co'] = expanding_count(data['campaign_id']) # 1 No
data['coupon_id_exp_co'] = expanding_count(data['coupon_id']) # 2 No
data['customer_id_exp_co'] = expanding_count(data['customer_id']) # 3 No
data['rented_count'] = data['customer_id'].map(feature(customer_demographics, 'customer_id','rented','sum')).\
fillna(0.07964084495607981) # 4 No
#  campaign_id based features
data['campaign_id_count'] = data['campaign_id'].map(data['campaign_id'].value_counts()) #  No
data['coupon_id_count'] = data['coupon_id'].map(data['coupon_id'].value_counts())#  No
data['customer_id_count'] = data['customer_id'].map(data['customer_id'].value_counts())#  No

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
#data['difference'] = (data['end_date'] - data['start_date']) / np.timedelta64(1, 'D')
data['end_date_month'] = data['end_date'].dt.month
data['end_date_dayofweek'] = data['end_date'].dt.dayofweek 
# data['end_date_dayofyear'] = data['end_date'].dt.dayofyear 
# data['end_date_days_in_month'] = data['end_date'].dt.days_in_month 
data['start_date_month'] = data['start_date'].dt.month
data['start_date_dayofweek'] = data['start_date'].dt.dayofweek 
# data['start_date_dayofyear'] = data['start_date'].dt.dayofyear 
# data['start_date_days_in_month'] = data['start_date'].dt.days_in_month 
# data['diff_dayofweek'] = data['end_date_dayofweek'] - data['start_date_dayofweek']
# data['diff_dayofyear'] = data['end_date_dayofyear'] - data['start_date_dayofyear']
 ```
 ![](./dis.png) 
 
 [`DAY - 1`](./Day-1)
 

 
| `Experiment name`  | `MODEL`  | `CV`  | `LB` |`script`|
| ----------- | ----------- |----------- |----------- |----------- |
| 10 fold lightgbm SKFold       |LightGbm       |0.823346822002498       |0.723262091750793       |[script](./Day-1/day_1_sub_1.py)|
| 10 fold lightgbm SKFold       |LightGbm       |0.90       |0.823638085449945       |[script](./Day-1/day-1-script-02.py)       |
| 10 fold lightgbm SKFold       |LightGbm       |0.9019       |0.820   |[script](./Day-1/day-1-script-03.py)       |
| 10 fold Catboost SKFold       |Catboost       |0.8844        |0.782251894526244   |[script](./Day-1/day-1-script-04.py)       |
| 10 fold LR SKFold |LogisticRegression|0.714118745672373|0.652779212300426 |[script](./Day-1/day-1-script-05.py)|
| 10 fold RFClassifier SKFold |RandomForestClassifier|0.826985519407373|0.749643707078551|[script](./Day-1/day-1-script-06.py)|
| 10 fold Neural Network SKFold|Neural Network|0.89|0.784766559598147 |[script](./Day-1/day-1-script-07.py)|


 [`DAY - 2`](./Day-2)
  

| `Experiment name`  | `MODEL`  | `CV`  | `LB` |`Note`|`Args`|
| ----------- | ----------- |----------- |----------- |----------- |----------- |
| 1      | Title       |Title       |Title       |Title       |Title       |
| 2      | Title       |Title       |Title       |Title       |Title       |


 [`DAY - 3`](./Day-3)
  

| `Experiment name`  | `MODEL`  | `CV`  | `LB` |`Note`|`Args`|
| ----------- | ----------- |----------- |----------- |----------- |----------- |
| 1      | Title       |Title       |Title       |Title       |Title       |
| 2      | Title       |Title       |Title       |Title       |Title       |


 [`DAY - 4`](./Day-4)
  

| `Experiment name`  | `MODEL`  | `CV`  | `LB` |`Note`|`Args`|
| ----------- | ----------- |----------- |----------- |----------- |----------- |
| 1      | Title       |Title       |Title       |Title       |Title       |
| 2      | Title       |Title       |Title       |Title       |Title       |


 [`DAY - 5`](./Day-5)
  

| `Experiment name`  | `MODEL`  | `CV`  | `LB` |`Note`|`Args`|
| ----------- | ----------- |----------- |----------- |----------- |----------- |
| 1      | Title       |Title       |Title       |Title       |Title       |
| 2      | Title       |Title       |Title       |Title       |Title       |


 [`DAY - 6`](./Day-6)
  

| `Experiment name`  | `MODEL`  | `CV`  | `LB` |`Note`|`Args`|
| ----------- | ----------- |----------- |----------- |----------- |----------- |
| 1      | Title       |Title       |Title       |Title       |Title       |
| 2      | Title       |Title       |Title       |Title       |Title       |


 [`DAY - 7`](./Day-7)
  

| `Experiment name`  | `MODEL`  | `CV`  | `LB` |`Note`|`Args`|
| ----------- | ----------- |----------- |----------- |----------- |----------- |
| 1      | Title       |Title       |Title       |Title       |Title       |
| 2      | Title       |Title       |Title       |Title       |Title       |


 [`DAY - 8`](./Day-8)
  

| `Experiment name`  | `MODEL`  | `CV`  | `LB` |`Note`|`Args`|
| ----------- | ----------- |----------- |----------- |----------- |----------- |
| 1      | Title       |Title       |Title       |Title       |Title       |
| 2      | Title       |Title       |Title       |Title       |Title       |


 [`DAY - 9`](./Day-9)
  

| `Experiment name`  | `MODEL`  | `CV`  | `LB` |`Note`|`Args`|
| ----------- | ----------- |----------- |----------- |----------- |----------- |
| 1      | Title       |Title       |Title       |Title       |Title       |
| 2      | Title       |Title       |Title       |Title       |Title       |
