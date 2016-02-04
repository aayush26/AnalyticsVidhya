################## LIBRARIES ###################
import pandas as pd
import numpy as np
from sklearn import linear_model

################ dataset load ###################
train_data_df = pd.read_csv('train.csv', header = False)
test_data_df = pd.read_csv('test.csv', header= False)

output = pd.Series(train_data_df.Item_Outlet_Sales, dtype="float")
train_data_df = train_data_df.drop('Item_Outlet_Sales',1)
train_data_df = train_data_df.drop('Item_Identifier',1)
train_data_df = train_data_df.drop('Outlet_Identifier',1)
train_data_df = train_data_df.fillna(0)
train_data_df = train_data_df.astype('float')

sub = test_data_df.Item_Identifier, test_data_df.Outlet_Identifier
test_data_df = test_data_df.drop('Item_Identifier',1)
test_data_df = test_data_df.drop('Outlet_Identifier',1)
test_data_df = test_data_df.fillna(0)
test_data_df = test_data_df.astype('float')

# clf = linear_model.LinearRegression()
clf = linear_model.Ridge(alpha=15,tol=0.0001)
# clf = linear_model.Lasso(alpha=15,tol=0.0001)
# clf = linear_model.ElasticNet(alpha=15,tol=0.0001, l1_ratio=0.9)
clf = clf.fit(train_data_df, output)
pred = clf.predict(test_data_df)

print len(pred)

f = open("submission.csv","w")
f.write("Item_Identifier,Outlet_Identifier,Item_Outlet_Sales\n")
for i in range(len(pred)):
	f.write(str(sub[0][i])+","+str(sub[1][i])+","+str(pred[i]))
	f.write("\n")