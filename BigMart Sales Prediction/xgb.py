################## LIBRARIES ###################
import pandas as pd
import numpy as np
import xgboost as xgb

################ dataset load ###################
train_data_df = pd.read_csv('train.csv', header = False)
test_data_df = pd.read_csv('test.csv', header= False)

output = pd.Series(train_data_df.Item_Outlet_Sales, dtype="category")
train_data_df = train_data_df.drop('Item_Outlet_Sales',1)
train_data_df = train_data_df.drop('Item_Identifier',1)
train_data_df = train_data_df.drop('Outlet_Identifier',1)
# train_data_df = train_data_df.fillna(0)
# print train_data_df.columns
for col in train_data_df.columns:
	train_data_df[col]=train_data_df[col].astype('category')
# train_data_df = train_data_df.astype('category')

sub = test_data_df.Item_Identifier, test_data_df.Outlet_Identifier
test_data_df = test_data_df.drop('Item_Identifier',1)
test_data_df = test_data_df.drop('Outlet_Identifier',1)
# test_data_df = test_data_df.fillna(0)
for col in test_data_df.columns:
	test_data_df[col]=test_data_df[col].astype('category')
# test_data_df = test_data_df.astype('category')

# print len(test_data_df)

train_data_df = np.array(train_data_df)
test_data_df = np.array(test_data_df)

xgb_tr = xgb.DMatrix(train_data_df, label=output, missing=np.NaN)
xgb_ts = xgb.DMatrix(test_data_df, missing=np.NaN)

############## Prediction model #################
param = {}
param['objective'] = 'reg:linear'
param['eta'] = 0.1
param['max_delta_step'] = 2000
param['max_depth'] = 5		#20
# # param['num_class'] = 2
# param['lambda']=0.0001
# param['subsample'] = 0.85
# # param['colsample_bytree'] = 1
# param['gamma'] = 0.1
param['min_child_weight'] = 100		#100
num_round = 96					#96

gbm = xgb.train(param,xgb_tr,num_round)
test_pred = gbm.predict(xgb_ts)

# print test_pred[0]

f = open("submission.csv","w")
f.write("Item_Identifier,Outlet_Identifier,Item_Outlet_Sales\n")
for i in range(len(test_pred)):
	f.write(str(sub[0][i])+","+str(sub[1][i])+","+str(test_pred[i]))
	f.write("\n")

