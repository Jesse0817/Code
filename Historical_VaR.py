#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import datetime as dt
from pandas_datareader import data as pdr
import yfinance as yf
yf.pdr_override()


# In[10]:


stocklist=['CBA', 'BHP', 'TLS', 'NAB', 'WBC', 'STO']
stocks=[stock + '.AX' for stock in stocklist]
endDate=dt.datetime.now()
startDate=endDate-dt.timedelta(days=800)


# In[31]:


returns, meanReturns, covMatrix = get_data(stocks, startDate, endDate)
returns = returns.dropna()

weights = np.random.random(len(returns.columns))
weights = weights/np.sum(weights)

returns['portfolio'] = returns.dot(weights)

VaR = historicalVaR(returns['portfolio'], alpha=5)
CVaR = historicalCVaR(returns['portfolio'], alpha=5)


# In[36]:


Time = 252

VaR = historicalVaR(returns['portfolio'], alpha=5) * np.sqrt(Time)
CVaR = historicalCVaR(returns['portfolio'], alpha=5) * np.sqrt(Time)

pRet, pStd = portfolioPerformance(weights, meanReturns, covMatrix, Time)

initialInvestment = 1000000
print('Expected portfolio returns: ', round(initialInvestment * pRet, 2))
print('VaR 95th CI               : ', round(initialInvestment * VaR, 2))
print('Conditional VaR 95th CI   : ', round(initialInvestment * CVaR, 2))


# In[12]:


def get_data(stocks, start, end):
    stockData=pdr.get_data_yahoo(stocks, start, end)
    stockData=stockData['Close']
    returns=stockData.pct_change()
    meanReturns=returns.mean()
    covMatrix=returns.cov()
    return returns, meanReturns, covMatrix


# In[5]:


def portfolioPerformance(weights, meanReturns, covMatrix, Time):
    returns = np.sum(meanReturns * weights) * Time
    std = np.sqrt(np.dot(weights.T, np.dot(covMatrix, weights))) * np.sqrt(Time)
    return returns, std


# In[24]:


def historicalVaR(returns, alpha=5):
    if isinstance(returns, pd.Series):
        return np.percentile(returns, alpha)
    elif isinstance(returns, pd.DataFrame):
        return returns.aggregate(historicalVaR, alpha=5)
    else:
        raise TypeError("Expected returns to be dataframe or series")
    


# In[28]:


def historicalCVaR(returns, alpha=5):
    if isinstance(returns, pd.Series):
        belowVaR = returns <= historicalVaR(returns, alpha=alpha)
        return returns[belowVaR].mean()
    elif isinstance(returns, pd.DataFrame):
        return returns.aggregate(historicalCVaR, alpha=5)
    else:
        raise TypeError("Expected returns to be dataframe or series")


# In[1]:


pip install pandas-datareader


# In[2]:


pip install yfinance


# In[ ]:




