# %matplotlib inline

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize']=15,6

dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m')
data = pd.read_csv('AirPassengers.csv', parse_dates='Month', index_col='Month', date_parser=dateparse)
# print data.head()
# print '\n Data Types:'
# print data.dtypes
# print data.index

ts = data['#Passengers']
# print ts.head(10)
# print ts['1949-01-01':'1949-05-01'] #end index is included unlike numerical [a,b]

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

# test_stationarity(ts)

# # ESTIMATING & ELIMINATING TREND
ts_log = np.log(ts)

#using rolling mean
# moving_avg = pd.rolling_mean(ts_log, 12)
# # plt.plot(ts_log, color="blue")
# # plt.plot(moving_avg, color="red")
# # plt.show()
# ts_log_moving_avg_diff = ts_log - moving_avg
# ts_log_moving_avg_diff.dropna(inplace=True)
# test_stationarity(ts_log_moving_avg_diff)

#using exponentially weighted moving average
expweighted_avg = pd.ewma(ts_log, halflife=12)
# # plt.plot(ts_log)
# # plt.plot(expweighted_avg, color="red")
# # plt.show()
# ts_log_ewma_diff = ts_log - expweighted_avg
# test_stationarity(ts_log_ewma_diff)

# # ELIMINATING TREND AND SEASONALITY
ts_log_diff = ts_log - ts_log.shift()
# # plt.plot(ts_log_diff)
# # plt.show()
ts_log_diff.dropna(inplace=True)
# test_stationarity(ts_log_diff)

from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(ts_log)

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

# plt.subplot(411)
# plt.plot(ts_log, label='Original')
# plt.legend(loc='best')
# plt.subplot(412)
# plt.plot(trend, label='Trend')
# plt.legend(loc='best')
# plt.subplot(413)
# plt.plot(seasonal,label='Seasonality')
# plt.legend(loc='best')
# plt.subplot(414)
# plt.plot(residual, label='Residuals')
# plt.legend(loc='best')
# plt.tight_layout()
# plt.show()

# model using only residual
ts_log_decompose = residual
# ts_log_decompose.dropna(inplace=True)
# test_stationarity(ts_log_decompose)

# # FORECASTING TIME SERIES
# #ACF and PACF plots:
# from statsmodels.tsa.stattools import acf, pacf

# lag_acf = acf(ts_log_diff, nlags=20)
# lag_pacf = pacf(ts_log_diff, nlags=20, method='ols')

# #Plot ACF: 
# # To derive q where the ACF chart crosses the upper confidence interval for the first time.
# plt.subplot(121) 
# plt.plot(lag_acf)
# plt.axhline(y=0,linestyle='--',color='gray')
# plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
# plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
# plt.title('Autocorrelation Function')

# #Plot PACF:
# #To derive p where the PACF chart crosses the upper confidence interval for the first time.
# plt.subplot(122)
# plt.plot(lag_pacf)
# plt.axhline(y=0,linestyle='--',color='gray')
# plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
# plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
# plt.title('Partial Autocorrelation Function')
# plt.tight_layout()

# plt.show()

# ARIMA Model
from statsmodels.tsa.arima_model import ARIMA, ARMAResults, ARMA

# model = ARIMA(ts_log, order=(2, 1, 0))  
# results_AR = model.fit(disp=-1)
# plt.plot(ts_log_diff)
# plt.plot(results_AR.fittedvalues, color='red')
# plt.title('RSS: %.4f'% sum((results_AR.fittedvalues-ts_log_diff)**2))

# model = ARIMA(ts_log, order=(0, 1, 2))  
# results_MA = model.fit(disp=-1) 
# plt.plot(ts_log_diff)
# plt.plot(results_MA.fittedvalues, color='red')
# plt.title('RSS: %.4f'% sum((results_MA.fittedvalues-ts_log_diff)**2))

model = ARIMA(ts_log, order=(2, 1, 2))  
results_ARIMA = model.fit(disp=-1)
plt.plot(ts_log_diff)
plt.plot(results_ARIMA.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_ARIMA.fittedvalues-ts_log_diff)**2))
plt.show()

# predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
# print predictions_ARIMA_diff.head()

# predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
# print predictions_ARIMA_diff_cumsum.head()

# predictions_ARIMA_log = pd.Series(ts_log.ix[0], index=ts_log.index)
# predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)
# print predictions_ARIMA_log.head()

# predictions_ARIMA = np.exp(predictions_ARIMA_log)
# plt.plot(ts)
# plt.plot(predictions_ARIMA)
# plt.title('RMSE: %.4f'% np.sqrt(sum((predictions_ARIMA-ts)**2)/len(ts)))
# plt.show()

# arma_mod30 = ARMA(ts_log , (4,1)).fit()
# # print ts_log
# # print arma_mod30
# predict_sunspots = arma_mod30.predict('1951-01-01', '1971-01-01', dynamic=True)
# # print ts_log.tail()
# predictions_ARIMA_diff = pd.Series(predict_sunspots, copy=True)
# # predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
# # predictions_ARIMA_log = pd.Series(ts_log.ix[0], index=ts_log.index)
# # predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff,fill_value=0)
# # print predictions_ARIMA_log.head()


# plt.plot(ts_log, color='red')
# plt.plot(predictions_ARIMA_diff, color='blue')
# plt.show()
