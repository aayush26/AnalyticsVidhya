################## LIBRARIES ###################
import pandas as pd
import numpy as np
import xgboost as xgb

################ dataset load ###################
train_data_df = pd.read_csv('train.csv', header = False)
test_data_df = pd.read_csv('test.csv', header= False)

output = pd.Series(train_data_df.Loan_Status, dtype="float")
train_data_df = train_data_df.drop('Loan_Status',1)
train_data_df = train_data_df.drop('Loan_ID',1)

id = test_data_df.Loan_ID
test_data_df = test_data_df.drop('Loan_ID',1)

test_data_df = test_data_df.astype('float')
train_data_df = train_data_df.astype('float')

# print len(test_data_df)

# train_data_df = np.array(train_data_df)
# test_data_df = np.array(test_data_df)

xgb_tr = xgb.DMatrix(train_data_df, label=output, missing=np.NaN)
xgb_ts = xgb.DMatrix(test_data_df, missing=np.NaN)

############## Prediction model #################
param = {}
param['objective'] = 'binary:logitraw'
param['eta'] = 0.1
param['max_delta_step'] = 1000
param['max_depth'] = 5		#20
# param['num_class'] = 2
param['lambda']=0.001
param['subsample'] = 0.85
# param['colsample_bytree'] = 1
param['gamma'] = 1
param['min_child_weight'] = 17
num_round = 2000					#2000

gbm = xgb.train(param,xgb_tr,num_round)
test_pred = gbm.predict(xgb_ts)

# print type(test_pred[0])

f = open("submission.csv","w")
f.write("Loan_ID,Loan_Status\n")
for i in range(len(test_pred)):
	if test_pred[i]<=0:
		a = "N"
	else:
		a = "Y"
	f.write(str(id[i])+","+a)
	f.write("\n")

