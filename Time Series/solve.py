import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize']=15,6

dateparse = lambda dates: pd.datetime.strptime(dates, '%d-%m-%Y %H:%M')
train_df = pd.read_csv('Train.csv', parse_dates='Datetime', index_col='Datetime', date_parser=dateparse)
test_df = open('Test.csv', 'r')
datetime_col = []
for lines in test_df.readlines():
	lines = lines.replace("\n","").replace("\r","")
	datetime_col.append(lines)
	# break
# print datetime_col

# print train_df.index

ts = train_df['Count']
# print ts.head()

# plt.plot(ts)
# plt.show()

#CHECK STATIONALITY
from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries):
	
	#Determing rolling statistics
	rolmean = pd.rolling_mean(timeseries, window=12)
	rolstd = pd.rolling_std(timeseries, window=12)

	#Plot rolling statistics:
	orig = plt.plot(timeseries, color='blue',label='Original')
	mean = plt.plot(rolmean, color='red', label='Rolling Mean')
	std = plt.plot(rolstd, color='black', label = 'Rolling Std')
	plt.legend(loc='best')
	plt.title('Rolling Mean & Standard Deviation')
	plt.show()
	
	#Perform Dickey-Fuller test:
	#null hypothesis: TS(Timestamp) is non-stationary
	#If test statistic < any Critical Value, reject null hypothesis, i.e. TS is stationary
	print 'Results of Dickey-Fuller Test:'
	dftest = adfuller(timeseries, autolag='AIC')
	dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
	for key,value in dftest[4].items():
		dfoutput['Critical Value (%s)'%key] = value
	print dfoutput


ts_log = np.log(ts)
# ts_log_diff = ts_log - ts_log.shift()
# ts_log_diff.dropna(inplace=True)
# test_stationarity(ts_log_diff)

from statsmodels.tsa.arima_model import ARIMA, ARMAResults, ARMA

arma_mod30 = ARMA(ts_log , (2,1)).fit()
# # print ts_log
# # print arma_mod30
predict_sunspots = arma_mod30.predict('26-09-2014 00:00', '26-04-2015 23:00', dynamic=True)
p_exp = np.exp(predict_sunspots)
print len(p_exp)

# from math import round

f = open("output.csv","w")
f.write("Datetime,Count\n")
for i in range(1,len(p_exp)):
	f.write(datetime_col[i-1] + "," + str(int(round(p_exp[i]))))
	f.write("\n")
f.write("26-04-2015 23:00,"+str(int(round(p_exp[5111]))))
f.write("\n")