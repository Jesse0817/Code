#import package

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import datetime as dt
from pandas_datareader import data as pdr
import yfinance as yf
yf.pdr_override()



#choosing stocks
#setting data

stocklist=['CBA', 'BHP', 'TLS', 'NAB', 'WBC', 'STO']
stocks=[stock + '.AX' for stock in stocklist]
endDate=dt.datetime.now()
startDate=endDate-dt.timedelta(days=800)



#calculating returns, weight, VaR, and Conditional VaR by fuctions 

returns, meanReturns, covMatrix = get_data(stocks, startDate, endDate)
returns = returns.dropna()

weights = np.random.random(len(returns.columns))
weights = weights/np.sum(weights)

returns['portfolio'] = returns.dot(weights)

VaR = historicalVaR(returns['portfolio'], alpha=5)
CVaR = historicalCVaR(returns['portfolio'], alpha=5)



#with windows (252 days)

Time = 252

VaR = historicalVaR(returns['portfolio'], alpha=5) * np.sqrt(Time)
CVaR = historicalCVaR(returns['portfolio'], alpha=5) * np.sqrt(Time)

pRet, pStd = portfolioPerformance(weights, meanReturns, covMatrix, Time)

initialInvestment = 1000000
print('Expected portfolio returns: ', round(initialInvestment * pRet, 2))
print('VaR 95th CI               : ', round(initialInvestment * VaR, 2))
print('Conditional VaR 95th CI   : ', round(initialInvestment * CVaR, 2))



#functions:
#getting returns, meanReturns, and covMatrix from yahoo

def get_data(stocks, start, end):
    stockData=pdr.get_data_yahoo(stocks, start, end)
    stockData=stockData['Close']
    returns=stockData.pct_change()
    meanReturns=returns.mean()
    covMatrix=returns.cov()
    return returns, meanReturns, covMatrix



#calculating returns and std from portfolio

def portfolioPerformance(weights, meanReturns, covMatrix, Time):
    returns = np.sum(meanReturns * weights) * Time
    std = np.sqrt(np.dot(weights.T, np.dot(covMatrix, weights))) * np.sqrt(Time)
    return returns, std



#calculating VaR with historical method

def historicalVaR(returns, alpha=5):
    if isinstance(returns, pd.Series):
        return np.percentile(returns, alpha)
    elif isinstance(returns, pd.DataFrame):
        return returns.aggregate(historicalVaR, alpha=5)
    else:
        raise TypeError("Expected returns to be dataframe or series")


        
#calculating Conditional VaR with historical method

def historicalCVaR(returns, alpha=5):
    if isinstance(returns, pd.Series):
        belowVaR = returns <= historicalVaR(returns, alpha=alpha)
        return returns[belowVaR].mean()
    elif isinstance(returns, pd.DataFrame):
        return returns.aggregate(historicalCVaR, alpha=5)
    else:
        raise TypeError("Expected returns to be dataframe or series")

#pip install:
pip install pandas-datareader
pip install yfinance
