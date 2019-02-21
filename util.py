from __future__ import division
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import *
from sklearn.multioutput import MultiOutputRegressor
#ACF and PACF plots:
from statsmodels.tsa.stattools import acf, pacf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import scipy as sp
from scipy.interpolate import spline
from sklearn.preprocessing import MinMaxScaler
import sklearn as sk # data mining tools
import matplotlib.pylab as plt # plotting
import seaborn as sns # advanced plotting
from pandas.tools.plotting import scatter_matrix
from matplotlib import rc
from matplotlib import ticker
import matplotlib as mpl
import matplotlib.colors as colors
#from geopy.geocoders import Nominatim
from collections import defaultdict
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import *
from sklearn.preprocessing import MinMaxScaler
from scipy.stats.stats import pearsonr
from operator import itemgetter
import math
import random
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.ensemble import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor

critic_lsoa = ['E01000855', 'E01000863', 'E01000914', 'E01000919', 'E01001043',
       'E01002444', 'E01002724', 'E01004714', 'E01004734', 'E01004735',
       'E01004736', 'E01004761', 'E01004763', 'E01004765', 'E01033595',
       'E01033708']

def print_sorted_scores(scores, sort_on = "r2", topk = 10, prints = True):
    errs = {
        'ev': 2,
        'mae': 4,
        'r2' : 6,
        'corr': 8,
    }
    reverse = True
    if sort_on == 'mae':
        reverse = False
    ss = sorted(scores, key=itemgetter(errs[sort_on]), reverse=reverse)
    l = min(topk, len(scores))
    if prints:
        for i in range(l):
            print ('-----------------------------')
            print ('Algorithm: ', ss[i][0], 'prev_obs = ', ss[i][1])
            print ('avg(ev) = %.3f std dev = %.3f' % (ss[i][2], ss[i][3]))
            print ('avg(mae) = %.3f std dev = %.3f' % (ss[i][4], ss[i][5]))
            print ('avg(r2) = %.3f std dev = %.3f' % (ss[i][6], ss[i][7]))
            if not(math.isnan(ss[i][8])):
                print ('avg(corr)= ', ss[i][8], 'std dev = ', ss[i][9])
            else:
                print ('avg(corr) = %.3f std dev = %.3f' % (ss[i][8], ss[i][9]))
            print ('Parameters -->')
            print (ss[i][10])
    return ss[:l];

def normalize_data(cd_s):
    data = cd_s.values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaler.fit(data)
    return scaler.transform(data).reshape(1, -1)[0], scaler

def get_chunks(ts, prev_obs, to_predict):
    x, y = [], []
    for i in range(prev_obs, len(ts)):
        x.append(ts[i-prev_obs:i])
        y.append(ts[i:i+to_predict])
    return x, y

"""
    Function that splits a time series into train and test
    and returns data in this format:
    
    [x_1 ... x_prev_obs ] [y_1 ... y_to_predict]

"""
def train_test(ts, ranges, prev_obs, to_predict):
    train = ts.loc[ranges['train'][0]:ranges['train'][1]].values
    test = ts.loc[ranges['test'][0]:ranges['test'][1]].values
    
    x_train, y_train = get_chunks(train, prev_obs, to_predict)
    x_test, y_test = get_chunks(test, prev_obs, to_predict)
    
    data = dict({
        "train" : {
            "x" : x_train,
            "y" : y_train
        },
        "test" : {
            "x" : x_test,
            "y" : y_test
        }
    })
    return data

def print_scores(true_, pred, ret_scores = False):
    ev, mae, r2, corr = explained_variance_score(true_, pred), mean_absolute_error(true_, pred), r2_score(true_, pred), pearsonr(true_, pred)[0]  
    if ret_scores:
        return [ev, mae, r2, corr]
    else:
        print ('explained variance ',ev)
        print ('MAE ',mae)
        print ('R2 ',r2)
        print ('corr ', corr)
        
def get_bor_ts2(series, b):
    west = series[series['Borough'] == b]
    west.month = pd.to_datetime(west.month)
    west = west.groupby(['month'], as_index=True).aggregate(np.sum)
    return west        

def get_bor_ts(series, b):
    west = series[series['borough'] == b]
    west.month = pd.to_datetime(west.month)
    west = west.groupby(['month'], as_index=True).aggregate(np.sum)
    return west

def get_lsoa_ts(series, b):
    west = series[series['lsoa_code'] == b]
    west.month = pd.to_datetime(west.month)
    west = west.groupby(['month'], as_index=True).aggregate(np.sum)
    return west

from sklearn.dummy import *

def performClassification(alg_str, x_trainFolds, y_trainFolds, x_testFold, y_testFold, parameters):
    y_pred = []
    if alg_str == 'random_forest':
        regr_rf = RandomForestRegressor(**parameters)
        regr_rf.fit(x_trainFolds, y_trainFolds)
        # Predict on new data
        #y_multirf = regr_multirf.predict()
        y_pred = regr_rf.predict(x_testFold)
        
    elif alg_str == "gradient_boost":
        clf = GradientBoostingRegressor(**parameters)
        clf.fit(x_trainFolds, y_trainFolds)
        y_pred = clf.predict(x_testFold)
    elif alg_str == 'arima':
        r1 = arima(x_trainFolds, (1, 0, 0), plot = False)[0]
        predictions_ARIMA_ = pd.Series(r1.fittedvalues, copy=True)
        pred = r1.predict(start=test_idx[0], end=test_idx[-1])
    elif alg_str == 'dummy_avg':
        dummy  = DummyRegressor(strategy='mean')
        dummy.fit(x_trainFolds, y_trainFolds)
        y_pred = dummy.predict(x_testFold)
        
    ss = print_scores(np.array(y_testFold).reshape(-1), np.array(y_pred).reshape(-1), ret_scores=True)
    
    return ss[0], ss[1], ss[2], ss[3]
    
def performTimeSeriesCV(X_train, y_train, number_folds, alg_str, parameters, print_all = False):
    """
    Given X_train and y_train (the test set is excluded from the Cross Validation),
    number of folds, the algorithm to implement and the parameters to test,
    the function acts based on the following logic: it splits X_train and y_train in a
    number of folds equal to number_folds. Then train on one fold and tests accuracy
    on the consecutive as follows:
    - Train on fold 1, test on 2
    - Train on fold 1-2, test on 3
    - Train on fold 1-2-3, test on 4
    ....
    Returns mean of test accuracies.

    print 'Algorithm ----> ', alg_str, ' | Parameters ----> ', parameters
    print 'Size train set: ', X_train.shape
    """
    # k is the size of each fold. It is computed dividing the number of 
    # rows in X_train by number_folds. This number is floored and coerced to int
    k = int(np.floor(float(X_train.shape[0]) / number_folds))
    #print 'Size of each fold: ', k
    
    # initialize to zero the accuracies array. It is important to stress that
    # in the CV of Time Series if I have n folds I test n-1 folds as the first
    # one is always needed to train
    acc = np.zeros(number_folds-1)
    acc1 = np.zeros(number_folds-1)
    acc2 = np.zeros(number_folds-1)
    acc3 = np.zeros(number_folds-1)
    # loop from the first 2 folds to the total number of folds    
    for i in range(2, number_folds + 1):
        if print_all:
            print ('')
        
        # the split is the percentage at which to split the folds into train
        # and test. For example when i = 2 we are taking the first 2 folds out 
        # of the total available. In this specific case we have to split the
        # two of them in half (train on the first, test on the second), 
        # so split = 1/2 = 0.5 = 50%. When i = 3 we are taking the first 3 folds 
        # out of the total available, meaning that we have to split the three of them
        # in two at split = 2/3 = 0.66 = 66% (train on the first 2 and test on the
        # following)
        split = float(i-1)/i
        
        # example with i = 4 (first 4 folds):
        #      Splitting the first       4        chunks at          3      /        4
        if print_all:
            print ('Splitting the first ' + str(i) + ' chunks at ' + str(i-1) + '/' + str(i) )
        
        # as we loop over the folds X and y are updated and increase in size.
        # This is the data that is going to be split and it increases in size 
        # in the loop as we account for more folds. If k = 300, with i starting from 2
        # the result is the following in the loop
        # i = 2
        # X = X_train[:(600)]
        # y = y_train[:(600)]
        #
        # i = 3
        # X = X_train[:(900)]
        # y = y_train[:(900)]
        # .... 
        X = X_train[:(k*i)]
        y = y_train[:(k*i)]
        if print_all:
            print ('Size of train + test: ', X.shape) # the size of the dataframe is going to be k*i

        # X and y contain both the folds to train and the fold to test.
        # index is the integer telling us where to split, according to the
        # split percentage we have set above
        index = int(np.floor(X.shape[0] * split))
        
        # folds used to train the model        
        x_trainFolds = X[:index]        
        y_trainFolds = y[:index]
        
        # fold used to test the model
        x_testFold = X[(index + 1):]
        y_testFold = y[(index + 1):]
        
        # i starts from 2 so the zeroth element in accuracies array is i-2. performClassification() is a function which takes care of a classification problem. This is only an example and you can replace this function with whatever ML approach you need.
        acc[i-2], acc1[i-2], acc2[i-2], acc3[i-2] = performClassification(alg_str, x_trainFolds, y_trainFolds, x_testFold, y_testFold, parameters)
        
        # example with i = 4:
        #      Accuracy on fold         4     :    0.85423
        if print_all:
            print ('ExpVar on fold ' + str(i) + ': ', acc[i-2])
            print ('MAE on fold ' + str(i) + ': ', acc1[i-2])
            print ('R2 on fold ' + str(i) + ': ', acc2[i-2])
            print ('PearsonCorr on fold ' + str(i) + ': ', acc3[i-2])
        
    if print_all:
        print ("---------------------------------")
        print ('avg(ev) = ', acc.mean(), " std dev = ", acc.std())
        print ('avg(mae) = ', acc1.mean(), " std dev = ", acc1.std())
        print ('avg(r2) = ', acc2.mean(), " std dev = ", acc2.std())
        print ('avg(corr) = ', acc3.mean(), " std dev = ", acc3.std())

    # the function returns the mean of the accuracy on the n-1 folds    
    return acc.mean(), acc.std(),acc1.mean(), acc1.std(),acc2.mean(), acc2.std(),acc3.mean(), acc3.std(), parameters;

def acf_pacf(ts_log_diff, ret_lags=False):
    lag_acf = acf(ts_log_diff, nlags=20)
    lag_pacf = pacf(ts_log_diff, nlags=20, method='ols')

    #Plot ACF: 
    plt.subplot(121) 
    plt.plot(lag_acf)
    plt.axhline(y=0,linestyle='--',color='gray')
    plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
    plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
    plt.title('Autocorrelation Function')

    #Plot PACF:
    plt.subplot(122)
    plt.plot(lag_pacf)
    plt.axhline(y=0,linestyle='--',color='gray')
    plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
    plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
    plt.title('Partial Autocorrelation Function')
    plt.tight_layout()
    plt.show()
    if ret_lags:
        return [lag_acf, lag_pacf]

def dummy_constant(value, test):
    res = []
    for i in range (0,len(test)):
        res.append(value)
    return np.array(res)

def dummy_avg(trainY, test):    
    res=[]
    sum_ = np.array(trainY).sum()
    avg = math.floor(sum_/len(trainY))
    
    if((sum_/len(trainY))-avg >=0.5):
        avg = avg+1
    
    for i in range (0,len(test)):
        res.append(avg)
    return np.array(res)

def dummy_range(frm, to, test):
    res=[]    
    for i in range (0,len(test)):
        res.append(random.randint(frm, to))
    return np.array(res)
    
def performClassification2(alg_str, x_trainFolds, y_trainFolds, x_testFold, y_testFold, parameters):
    y_pred = []
    if alg_str == 'random_forest':
        regr_rf = RandomForestClassifier(**parameters)
        regr_rf.fit(x_trainFolds, y_trainFolds)
        # Predict on new data
        #y_multirf = regr_multirf.predict()
        y_pred = regr_rf.predict(x_testFold)
        
    elif alg_str == "gradient_boost":
        clf = GradientBoostingClassifier(**parameters)
        clf.fit(x_trainFolds, y_trainFolds)
        y_pred = clf.predict(x_testFold)
    elif alg_str == 'dummy_avg':
        dummy  = DummyRegressor(strategy='mean')
        dummy.fit(x_trainFolds, y_trainFolds)
        y_pred = dummy.predict(x_testFold)
    elif alg_str == 'dummy_avg':
        y_pred = dummy_avg(y_trainFolds, y_testFold).reshape(-1)
    elif alg_str == 'dummy_rand':
        y_pred = dummy_range(1,5, y_testFold).reshape(-1)
        
    ss = print_scores(np.array(y_testFold).reshape(-1), y_pred.reshape(-1), ret_scores=True)
    return ss[0], ss[1], ss[2], ss[3]
    
def performTimeSeriesCV2(X_train, y_train, number_folds, alg_str, parameters, print_all = False):
    """
    Given X_train and y_train (the test set is excluded from the Cross Validation),
    number of folds, the algorithm to implement and the parameters to test,
    the function acts based on the following logic: it splits X_train and y_train in a
    number of folds equal to number_folds. Then train on one fold and tests accuracy
    on the consecutive as follows:
    - Train on fold 1, test on 2
    - Train on fold 1-2, test on 3
    - Train on fold 1-2-3, test on 4
    ....
    Returns mean of test accuracies.

    print ('Algorithm ----> ', alg_str, ' | Parameters ----> ', parameters)
    print ('Size train set: ', X_train.shape)
    """
    # k is the size of each fold. It is computed dividing the number of 
    # rows in X_train by number_folds. This number is floored and coerced to int
    k = int(np.floor(float(X_train.shape[0]) / number_folds))
    #print 'Size of each fold: ', k
    
    # initialize to zero the accuracies array. It is important to stress that
    # in the CV of Time Series if I have n folds I test n-1 folds as the first
    # one is always needed to train
    acc = np.zeros(number_folds-1)
    acc1 = np.zeros(number_folds-1)
    acc2 = np.zeros(number_folds-1)
    acc3 = np.zeros(number_folds-1)
    # loop from the first 2 folds to the total number of folds    
    for i in range(2, number_folds + 1):
        if print_all:
            print ('')
        
        # the split is the percentage at which to split the folds into train
        # and test. For example when i = 2 we are taking the first 2 folds out 
        # of the total available. In this specific case we have to split the
        # two of them in half (train on the first, test on the second), 
        # so split = 1/2 = 0.5 = 50%. When i = 3 we are taking the first 3 folds 
        # out of the total available, meaning that we have to split the three of them
        # in two at split = 2/3 = 0.66 = 66% (train on the first 2 and test on the
        # following)
        split = float(i-1)/i
        
        # example with i = 4 (first 4 folds):
        #      Splitting the first       4        chunks at          3      /        4
        if print_all:
            print ('Splitting the first ' + str(i) + ' chunks at ' + str(i-1) + '/' + str(i) )
        
        # as we loop over the folds X and y are updated and increase in size.
        # This is the data that is going to be split and it increases in size 
        # in the loop as we account for more folds. If k = 300, with i starting from 2
        # the result is the following in the loop
        # i = 2
        # X = X_train[:(600)]
        # y = y_train[:(600)]
        #
        # i = 3
        # X = X_train[:(900)]
        # y = y_train[:(900)]
        # .... 
        X = X_train[:(k*i)]
        y = y_train[:(k*i)]
        if print_all:
            print ('Size of train + test: ', X.shape) # the size of the dataframe is going to be k*i)

        # X and y contain both the folds to train and the fold to test.
        # index is the integer telling us where to split, according to the
        # split percentage we have set above
        index = int(np.floor(X.shape[0] * split))
        
        # folds used to train the model        
        x_trainFolds = X[:index]        
        y_trainFolds = y[:index]
        
        # fold used to test the model
        x_testFold = X[(index + 1):]
        y_testFold = y[(index + 1):]
        
        # i starts from 2 so the zeroth element in accuracies array is i-2. performClassification() is a function which takes care of a classification problem. This is only an example and you can replace this function with whatever ML approach you need.
        acc[i-2], acc1[i-2], acc2[i-2], acc3[i-2] = performClassification2(alg_str, x_trainFolds, y_trainFolds, x_testFold, y_testFold, parameters)
        
        # example with i = 4:
        #      Accuracy on fold         4     :    0.85423
        if print_all:
           # print ('ExpVar on fold ' + str(i) + ': ', acc[i-2])
            print ('MAE on fold ' + str(i) + ': ', acc1[i-2])
            #print ('R2 on fold ' + str(i) + ': ', acc2[i-2])
            #print ('PearsonCorr on fold ' + str(i) + ': ', acc3[i-2])
        
 
    print ("---------------------------------")
    #print 'avg(ev) = ', acc.mean(), " std dev = ", acc.std()
    print ('avg(mae) = ', acc1.mean(), " std dev = ", acc1.std())
    #print 'avg(r2) = ', acc2.mean(), " std dev = ", acc2.std()
   # print 'avg(corr) = ', acc3.mean(), " std dev = ", acc3.std()
    # the function returns the mean of the accuracy on the n-1 folds    
    return acc.mean(), acc.std(),acc1.mean(), acc1.std(),acc2.mean(), acc2.std(),acc3.mean(), acc3.std(), parameters;
        
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)



def lstm(data,neurons,look_back,ep,batch,ver,plot=False):
    # load the dataset
    dataset = data.values.reshape(-1,1)
    dataset = dataset.astype('float32')

    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    # split into train and test sets
    train_size = int(len(dataset) * 0.67)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

    # reshape into X=t and Y=t+1
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)

    # reshape input to be [samples, time steps, features]
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

    # create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(neurons, input_shape=(1, look_back)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=ep, batch_size=batch, verbose=ver)

    # make predictions
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)

    print('Obs: ', look_back)
    ev=explained_variance_score(testY,testPredict)
    mae=mean_absolute_error(testY,testPredict)
    r2=r2_score(testY, testPredict)
    corr=pearsonr(testY,testPredict.reshape(-1))[0]

    # invert predictions
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])

    # shift train predictions for plotting
    trainPredictPlot = np.empty_like(dataset)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

    # shift test predictions for plotting
    testPredictPlot = np.empty_like(dataset)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
    
    if plot:
        # plot baseline and predictions
        plt.plot(scaler.inverse_transform(dataset))
        plt.plot(trainPredictPlot)
        plt.plot(testPredictPlot)
        plt.show()
    return [ev,mae,r2,corr]
	
def plot_lstm(scores, pos, label, title):
    y=[]
    
    x=np.arange(2,13,1)
    
    for i in range(0,len(scores)):
         y.append(scores[i][pos])
        
    #Plotting
    fig = plt.figure(figsize=(10,6))
    avg=np.array(y).mean()
    plt.xticks(rotation=45)
    plt.grid(color='grey', linestyle='--', linewidth=0.5)
    plt.title(title)
    #avg line
    plt.plot((x[0], x[-1]), (avg, avg), 'grey', label = "AVG")   
    
    plt.plot(x,y, label = label, color="blue")

    plt.legend()
  