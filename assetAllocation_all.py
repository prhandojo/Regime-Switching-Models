# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 16:43:35 2020

@author: prhandojo

library of functions for regime-based asset allocations: 
    
    markovRegSwitch(theta, data, m)
        To compute filtered regime probabilities and log-likelihood based on
        hidden markov chain regime-switching model
        
        Parameters:
            theta : array of regime switching model parameters. 
                    For m = 2 regime and for 2 assets(JCI and Bond), theta is structured as 
                    np.array([mu_JCI_1, mu_Bond_1, var_JCI_1, var_Bond_1, covar_1, p_1,
                              mu_JCI_2, mu_Bond_2, var_JCI_2, var_bond_2, covar_2, p_2])
                    where '_1' and '_2' indicates parameters for regime 1 and 2, respectively
            data : in-sample data
            m : number of regimes
    
    utilLog(w, ExRet, sigmaSq, vcov, Lambda)
        to compute utility based on log-returns and power utility function
        
        Parameters:
            w : array of assets' weights
            ExRet : array of assets' expected excess log-returns
            sigmaSq : array of variances of assets' excess log-returns
            vcov : variance-covariance matrix of assets' excess log-returns 
            Lambda : risk-aversion coefficient
        
    utilLogTC(w, w_prev, askArray, bidArray, ExRet, sigmaSq, vcov, Lambda)
        to compute utility based on log returns and power utility function
        with transaction costs included in the expected return calculation
        
        Parameters:
            w : array of assets' weights
            w_prev : array of assets' weights in the previous period
            askArray : array of unit fees for selling the assets
            bidArray : array of unit fees for buying the assets
            ExRet : array of assets' expected excess log-returns
            sigmaSq : array of variances of assets' excess log-returns
            vcov : variance-covariance matrix of assets' excess log-returns 
            Lambda : risk-aversion coefficient

    portCumRetTC(w, testData=validation) 
        to compute actual portfolio cumulative returns and asset weights 
        after transaction costs
        
        Parameters:
            w : array of assets' weights
            testData : data frame for which the model is to be applied, i.e. 
            validation or test data set
        
    portCumRetTC_monthly(w, testData=validation) 
        to compute actual portfolio cumulative returns and asset weights 
        after transaction costs with monthly rebalancing
        
        Parameters:
            w : array of assets' weights
            testData : data frame for which the model is to be applied, i.e. 
            validation or test data set
    
    MVE(Lambda)
        to compute asset weights using standard Mean-Variance algorithm
    
    MVE_TC(Lambda)
        to compute asset weights using Mean-Variance algorithm with TC
    
    regSw2(Lambda)
        to compute asset weights using regime switching (2 regime) model

    regSw2Kalman(Lambda)
        to compute asset weights using regime switching (2 regime) model
        and Kalman Filter to smoothen expected return and reduce asset weights
        fluctuations

    regSw2BLrel(optimVar)
        to compute asset weights using regime switching (2 regime) + 
        Black-Litterman model with relative view portfolio 
        
        Parameters:
            optimVar : np.array([Lambda, tau])

    regSw2BLabs(optimVar)
        to compute asset weights using regime switching (2 regime) + 
        Black-Litterman model with absolute view portfolio 
        
        Parameters:
            optimVar : np.array([Lambda, tau])

    regSw2_TC(Lambda)
        to compute asset weights using regime switching (2 regime) model and
        Mean-Variance algorithm with TC

    regSw2BLabs_TC(optimVar)
        to compute asset weights using regime switching (2 regime) + 
        Black-Litterman model with absolute view portfolio and
        Mean-Variance algorithm with TC
        
        Parameters:
            optimVar : np.array([Lambda, tau])

    regSw4(Lambda)
        to compute asset weights using regime switching (4 regime) model

    regSw4Kalman(Lambda)
        to compute asset weights using regime switching (4 regime) model
        and Kalman Filter to smoothen expected return and reduce asset weights
        fluctuations

    regSw4BLrel(optimVar)
        to compute asset weights using regime switching (4 regime) + 
        Black-Litterman model with relative view portfolio 
        
        Parameters:
            optimVar : np.array([Lambda, tau])

    regSw4BLabs(optimVar)
        to compute asset weights using regime switching (4 regime) + 
        Black-Litterman model with absolute view portfolio and
        Mean-Variance algorithm with TC
        
        Parameters:
            optimVar : np.array([Lambda, tau])

    regSw4_TC(Lambda)
        to compute asset weights using regime switching (4 regime) model and
        Mean-Variance algorithm with TC

    regSw4BLabs_TC(optimVar)
        to compute asset weights using regime switching (4 regime) + 
        Black-Litterman model with absolute view portfolio and
        Mean-Variance algorithm with TC
        
        Parameters:
            optimVar : np.array([Lambda, tau])

    heatmap(data, row_labels, col_labels, ax=None,
                cbar_kw={}, cbarlabel="", **kwargs)
        to plot heatmap of a 2D array data

    annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                         textcolors=("black", "white"),
                         threshold=None, **textkw)
        to annotate the heatmap using the data values
    
"""
import pandas as pd 
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy import optimize
from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint


# function to compute filtered regime probabilities and likelihood according to markov regime switching model
def markovRegSwitch(theta, data, m) :
    
    nPar = int(len(theta)/m)

    mu = [] 
    sig = []
    p = np.zeros((m,m))    
    colname=[]
    for i in range(0,m):
        # mean and variance-covariance matrix for each regime
        mu.append(theta[nPar*i:nPar*i+2])
        sig.append(np.matrix([[theta[nPar*i+2], theta[nPar*i+4]], 
                              [theta[nPar*i+4], theta[nPar*i+3]]]))
        
        # regime probability transition matrix
        for j in range(0,m-1):
            p[i][j]=theta[nPar*i+5+j]
        p[i][(m-1)]=1-sum(p[i][0:(m-1)])
        
        colname.append('p'+str(i+1))

    # prior about the filter, use equal probability for all regimes
    xi = np.matrix(np.repeat(1/m, m))

    Xi = pd.DataFrame(columns=colname) # to store filtered probabilities
    Lik = pd.DataFrame(columns=['likelihood']) # to store likelihood

    for t in range(0,len(data)) :
        phi = [] # likelihood given data on time t
        
        # compute likelihood for each regime 
        for i in range(0,m):
            phi.append(multivariate_normal.pdf(data.iloc[t,:], mean=mu[i], cov=sig[i]))
        phi = np.matrix(phi)
        
        # compute regime probability at time t, given t-1 regime probability and transition probabilities
        xi_next=np.transpose(p)*np.transpose(xi)
        
        # compute likelihood at time t
        Lik_t = phi*xi_next
        Lik.loc[t,'likelihood'] = Lik_t
        
        # update filter for next period
        xi = np.multiply(phi,np.transpose(xi_next))/Lik_t
        Xi.loc[t,] = xi
    
    # compute log-likelihood
    logLik = sum(np.log(Lik['likelihood']))
    return [logLik, Xi]


# utility function for Mean-Variance Efficient portfolio, based on log return data
def utilLog (w, ExRet, sigmaSq, vcov, Lambda) :
    u = np.matmul(w, ExRet) + 0.5*np.matmul(w,sigmaSq) - \
        0.5*np.matmul(np.matmul(w, vcov), np.transpose(w)) + \
            0.5*(1-Lambda)*np.matmul(np.matmul(w, vcov), np.transpose(w))
    return -float(u)


# utility function for Mean-Variance Efficient portfolio, based on log return data
# and including transaction costs in the calculation of expected returns
def utilLogTC (w, w_prev, askArray, bidArray, ExRet, sigmaSq, vcov, Lambda) :
    # compute assets weights to be bought 
    w_plus = np.fmax((w-w_prev),0)
    # compute assets weights to be sold 
    w_minus = np.fmax((w_prev-w),0)
    # compute cost of rebalancing
    cost = np.matmul(w_plus.T, bidArray) + np.matmul(w_minus.T, askArray)
    
    u = np.log(1 - cost) + np.matmul(w, ExRet)/(1-cost) + \
        0.5*np.matmul(w,sigmaSq)/(1-cost) - \
            0.5*np.matmul(np.matmul(w, vcov), np.transpose(w))/(1-cost)**2 + \
                0.5*(1-Lambda)*np.matmul(np.matmul(w, vcov), np.transpose(w))
    return -float(u)


# function to compute portfolio value and cumulative returns after transaction cost 
def portCumRetTC (w, testData=validation) : 
    W0 = 100 # set arbitraty initial wealth to calculate costs
    
    assetVal = pd.DataFrame(columns=['JCI', 'Bond', 'cash'])  
    w_actual = pd.DataFrame(columns=['JCI', 'Bond', 'cash'])
    TC_JCI = 0
    TC_Bond = 0
    ret = testData[['JCIWeeklyRet', 'BondWeeklyRet', "Rf_weekly"]]  

    # start at beginning of week 0 by buying the required assets and substract TC
    # TC is included in the total value of assets to be bought/sold, 
    # so the actual value of assets bought/sold are less than the value required from the MVE weights
    assetVal.loc[0, 'JCI'] = W0*w.loc[0, 'JCI']/(1+bid)
    assetVal.loc[0, 'Bond'] = W0*w.loc[0, 'Bond']/(1+bid)
    assetVal.loc[0, 'cash'] = W0*(1- w.loc[0, 'JCI'] - w.loc[0, 'Bond']) 

    # compute actual weights
    w_actual.loc[0, 'JCI'] = assetVal.loc[0, 'JCI']/sum(assetVal.loc[0])
    w_actual.loc[0, 'Bond'] = assetVal.loc[0, 'Bond']/sum(assetVal.loc[0])
    w_actual.loc[0, 'cash'] = assetVal.loc[0, 'cash']/sum(assetVal.loc[0])

    # compute the asset value at the end of week 0 by multiplying with the weekly returns
    assetVal['JCI'].iloc[0] = assetVal['JCI'].iloc[0]*(1+ret['JCIWeeklyRet'].iloc[0])
    assetVal['Bond'].iloc[0] = assetVal['Bond'].iloc[0]*(1+ret['BondWeeklyRet'].iloc[0])
    assetVal['cash'].iloc[0] = assetVal['cash'].iloc[0]*(1+ret['Rf_weekly'].iloc[0])

    for t in range(1,len(testData)) :   
        # rebalance the portfolio to desired weights (from MVE)
        # compute required asset value, based on the required weights 
        JCIVal = w.loc[t, 'JCI']*sum(assetVal.loc[t-1])
        BondVal = w.loc[t, 'Bond']*sum(assetVal.loc[t-1])
    
        # compute amount to be rebalanced
        # positive delta indicates buy, negative delta indicates sell
        deltaJCI = JCIVal - assetVal.loc[t-1, 'JCI']
        deltaBond = BondVal - assetVal.loc[t-1, 'Bond']
        
        # before transactions, balance of cash account this week is the same as last week
        cashVal = assetVal.loc[t-1, 'cash']
    
        if(deltaJCI>=0) :
           assetVal.loc[t, 'JCI'] = assetVal.loc[t-1, 'JCI'] + deltaJCI/(1+bid)
           cashVal = cashVal - deltaJCI
           TC_JCI = TC_JCI + deltaJCI*(1 - 1/(1+bid))
        elif(deltaJCI<0) :
           assetVal.loc[t, 'JCI'] = assetVal.loc[t-1, 'JCI'] + deltaJCI
           cashVal = cashVal + abs(deltaJCI) - abs(deltaJCI)/(1+ask)*ask
           TC_JCI = TC_JCI + abs(deltaJCI)/(1+ask)*ask
        
        if(deltaBond>=0) :
            assetVal.loc[t, 'Bond'] = assetVal.loc[t-1, 'Bond'] + deltaBond/(1+bid)
            cashVal = cashVal - deltaBond
            TC_Bond = TC_Bond + deltaBond*(1 - 1/(1+bid))
        elif(deltaBond<0) :
            assetVal.loc[t, 'Bond'] = assetVal.loc[t-1, 'Bond'] + deltaBond
            cashVal = cashVal + abs(deltaBond) - abs(deltaBond)/(1+ask)*ask
            TC_Bond = TC_Bond + abs(deltaBond)/(1+ask)*ask
        
        assetVal.loc[t, 'cash'] = cashVal
    
        # compute actual weights
        w_actual.loc[t, 'JCI'] = assetVal.loc[t, 'JCI']/sum(assetVal.loc[t])
        w_actual.loc[t, 'Bond'] = assetVal.loc[t, 'Bond']/sum(assetVal.loc[t])
        w_actual.loc[t, 'cash'] = assetVal.loc[t, 'cash']/sum(assetVal.loc[t])
        
        # compute the asset value at the end of week t
        assetVal['JCI'].iloc[t] = assetVal['JCI'].iloc[t]*(1+ret['JCIWeeklyRet'].iloc[t])
        assetVal['Bond'].iloc[t] = assetVal['Bond'].iloc[t]*(1+ret['BondWeeklyRet'].iloc[t])
        assetVal['cash'].iloc[t] = assetVal['cash'].iloc[t]*(1+ret['Rf_weekly'].iloc[t])
        
    # compute portfolio value and returns
    portVal = assetVal['JCI'] + assetVal['Bond'] + assetVal['cash']
    lagPortVal = portVal.shift(1)
    lagPortVal[0] = W0
    portRet = portVal/lagPortVal - 1
    portExRet = portRet - testData['Rf_weekly'].reset_index(drop=True)
    portCumRet = np.cumprod(1+portRet)
    SR = np.mean(portExRet)*np.sqrt(52)/np.std(portRet)
    
    return [portCumRet, SR, assetVal, w_actual, TC_JCI, TC_Bond]


# function to compute portfolio value and cumulative returns after transaction cost 
# with monthly rebalancing
def portCumRetTC_monthly (w, testData=validation) : 
    W0 = 100 # set arbitraty initial wealth to calculate costs
    
    assetVal = pd.DataFrame(columns=['JCI', 'Bond', 'cash'])  
    w_actual = pd.DataFrame(columns=['JCI', 'Bond', 'cash'])
    TC_JCI = 0
    TC_Bond = 0
    ret = testData[['JCIWeeklyRet', 'BondWeeklyRet', "Rf_weekly"]]  

    # start at beginning of week 0 by buying the required assets and substract TC
    # TC is included in the total value of assets to be bought/sold, 
    # so the actual value of assets bought/sold are less than the value required from the MVE weights
    assetVal.loc[0, 'JCI'] = W0*w.loc[0, 'JCI']/(1+bid)
    assetVal.loc[0, 'Bond'] = W0*w.loc[0, 'Bond']/(1+bid)
    assetVal.loc[0, 'cash'] = W0*(1- w.loc[0, 'JCI'] - w.loc[0, 'Bond']) 

    # compute actual weights
    w_actual.loc[0, 'JCI'] = assetVal.loc[0, 'JCI']/sum(assetVal.loc[0])
    w_actual.loc[0, 'Bond'] = assetVal.loc[0, 'Bond']/sum(assetVal.loc[0])
    w_actual.loc[0, 'cash'] = assetVal.loc[0, 'cash']/sum(assetVal.loc[0])

    # compute the asset value at the end of week 0 by multiplying with the weekly returns
    assetVal['JCI'].iloc[0] = assetVal['JCI'].iloc[0]*(1+ret['JCIWeeklyRet'].iloc[0])
    assetVal['Bond'].iloc[0] = assetVal['Bond'].iloc[0]*(1+ret['BondWeeklyRet'].iloc[0])
    assetVal['cash'].iloc[0] = assetVal['cash'].iloc[0]*(1+ret['Rf_weekly'].iloc[0])

    for t in range(1,len(testData)) :   
        # rebalance the portfolio to the desired weights (from MVE) only for monthly period (4 weeks)
        if(t%4==0) : 
            # rebalance the portfolio to desired weights (from MVE)
            # compute required asset value, based on the required weights 
            JCIVal = w.loc[t, 'JCI']*sum(assetVal.loc[t-1])
            BondVal = w.loc[t, 'Bond']*sum(assetVal.loc[t-1])
    
            # compute amount to be rebalanced
            # positive delta indicates buy, negative delta indicates sell
            deltaJCI = JCIVal - assetVal.loc[t-1, 'JCI']
            deltaBond = BondVal - assetVal.loc[t-1, 'Bond']
        
            # before transactions, balance of cash account this week is the same as last week
            cashVal = assetVal.loc[t-1, 'cash']
    
            if(deltaJCI>=0) :
                assetVal.loc[t, 'JCI'] = assetVal.loc[t-1, 'JCI'] + deltaJCI/(1+bid)
                cashVal = cashVal - deltaJCI
                TC_JCI = TC_JCI + deltaJCI*(1 - 1/(1+bid))
            elif(deltaJCI<0) :
                assetVal.loc[t, 'JCI'] = assetVal.loc[t-1, 'JCI'] + deltaJCI
                cashVal = cashVal + abs(deltaJCI) - abs(deltaJCI)/(1+ask)*ask
                TC_JCI = TC_JCI + abs(deltaJCI)/(1+ask)*ask
        
            if(deltaBond>=0) :
                assetVal.loc[t, 'Bond'] = assetVal.loc[t-1, 'Bond'] + deltaBond/(1+bid)
                cashVal = cashVal - deltaBond
                TC_Bond = TC_Bond + deltaBond*(1 - 1/(1+bid))
            elif(deltaBond<0) :
                assetVal.loc[t, 'Bond'] = assetVal.loc[t-1, 'Bond'] + deltaBond
                cashVal = cashVal + abs(deltaBond) - abs(deltaBond)/(1+ask)*ask
                TC_Bond = TC_Bond + abs(deltaBond)/(1+ask)*ask
        
            assetVal.loc[t, 'cash'] = cashVal
    
            # compute actual weights
            w_actual.loc[t, 'JCI'] = assetVal.loc[t, 'JCI']/sum(assetVal.loc[t])
            w_actual.loc[t, 'Bond'] = assetVal.loc[t, 'Bond']/sum(assetVal.loc[t])
            w_actual.loc[t, 'cash'] = assetVal.loc[t, 'cash']/sum(assetVal.loc[t])
        
            # compute the asset value at the end of week t
            assetVal['JCI'].iloc[t] = assetVal['JCI'].iloc[t]*(1+ret['JCIWeeklyRet'].iloc[t])
            assetVal['Bond'].iloc[t] = assetVal['Bond'].iloc[t]*(1+ret['BondWeeklyRet'].iloc[t])
            assetVal['cash'].iloc[t] = assetVal['cash'].iloc[t]*(1+ret['Rf_weekly'].iloc[t])
            
        elif(t%4!=0) : 
            # compute actual weights
            w_actual.loc[t, 'JCI'] = assetVal.loc[t-1, 'JCI']/sum(assetVal.loc[t-1])
            w_actual.loc[t, 'Bond'] = assetVal.loc[t-1, 'Bond']/sum(assetVal.loc[t-1])
            w_actual.loc[t, 'cash'] = assetVal.loc[t-1, 'cash']/sum(assetVal.loc[t-1])
            
            # compute the asset value at the end of week t
            assetVal.loc[t, 'JCI'] = assetVal['JCI'].iloc[t-1]*(1+ret['JCIWeeklyRet'].iloc[t])
            assetVal.loc[t, 'Bond'] = assetVal['Bond'].iloc[t-1]*(1+ret['BondWeeklyRet'].iloc[t])
            assetVal.loc[t, 'cash'] = assetVal['cash'].iloc[t-1]*(1+ret['Rf_weekly'].iloc[t])
        
        
    # compute portfolio value and returns
    portVal = assetVal['JCI'] + assetVal['Bond'] + assetVal['cash']
    lagPortVal = portVal.shift(1)
    lagPortVal[0] = W0
    portRet = portVal/lagPortVal - 1
    portExRet = portRet - testData['Rf_weekly'].reset_index(drop=True)
    portCumRet = np.cumprod(1+portRet)
    SR = np.mean(portExRet)*np.sqrt(52)/np.std(portRet)
    
    return [portCumRet, SR, assetVal, w_actual, TC_JCI, TC_Bond]


# Function to compute portfolio weights based on Mean-Variance Efficient Portfolio 
def MVE(Lambda) : 
    #-------------------------------------------#
    # Asset allocation for benchmark MVE model  #
    #-------------------------------------------#
    # compute log returns 
    data = pd.DataFrame()
    data['JCILogRet'] = np.log(1+train['JCIExRet'])
    data['bondLogRet'] = np.log(1+train['BondExRet'])
    w = pd.DataFrame(columns=['JCI', 'Bond'])
    
    for t in range(0,len(validation)) :
        # compute assets expected returns, variance-covariance matrix and 
        # vector of variances using all historical data up to last week
        ExRet = np.array([np.mean(data['JCILogRet']),
                          np.mean(data['bondLogRet'])])
        vcov = np.cov(data['JCILogRet'], data['bondLogRet'])
        sigmaSq = np.diag(vcov)
        
        # set bounds for MVE optimisation (no shorting and total weights = 0.95)
        # set aside 5% of cash to pay transaction costs
        bounds = Bounds([0, 0], [0.95,0.95])
        linear_constraint = LinearConstraint([1, 1], [0], [0.95])
        
        # set initial guess of the weights
        w0=np.array([0.4, 0.4]) 
        # run constraint-bound optimisation to determine assets weight
        result = optimize.minimize(fun = utilLog, 
                                   x0 = w0, 
                                   args = (ExRet, sigmaSq, vcov, Lambda),
                                   method = 'trust-constr', 
                                   bounds = bounds,
                                   constraints = linear_constraint,
                                   options = {'disp': False})
        w = w.append(pd.Series(result.x, index=w.columns), ignore_index=True)
        
        # update data for next period    
        data = data.append(pd.Series([np.log(1+validation['JCIExRet'].iloc[t]),
                                      np.log(1+validation['BondExRet'].iloc[t])], index=data.columns), 
                           ignore_index=True)
    return w


# Function to compute portfolio weights based on Mean-Variance Efficient Portfolio 
# with Transaction Cost in the MVO algorithm
def MVE_TC(Lambda) : 
    #----------------------------------------------------------------------#
    # Asset allocation for benchmark MVE model including transaction costs #
    #----------------------------------------------------------------------#

    # compute log returns 
    data = pd.DataFrame()
    data['JCILogRet'] = np.log(1+train['JCIExRet'])
    data['bondLogRet'] = np.log(1+train['BondExRet'])

    w_TC = pd.DataFrame(columns=['JCI', 'Bond'])

    # set weights at t=0 to be 0
    w_prev = np.array([0, 0]) 
    # array of bid and ask costs, assuming same unit costs for all assets
    askArray = np.array([ask, ask])
    bidArray = np.array([bid, bid])

    for t in range(0,len(validation)) :
        # compute assets expected returns, variance-covariance matrix and 
        # vector of variances using all historical data up to last week
        ExRet = np.array([np.mean(data['JCILogRet']),
                          np.mean(data['bondLogRet'])])
        vcov = np.cov(data['JCILogRet'], data['bondLogRet'])
        sigmaSq = np.diag(vcov)
    
        # set bounds for MVE optimisation (no shorting and total weights = 0.95)
        # set aside 5% of cash to pay transaction costs
        bounds = Bounds([0, 0], [0.95,0.95])
        linear_constraint = LinearConstraint([1, 1], [0], [0.95])
    
        # set initial guess of the weights
        w0=np.array([0.4, 0.4]) 
        # run constraint-bound optimisation to determine assets weight
        result = optimize.minimize(fun = utilLogTC, 
                                   x0 = w0, 
                                   args = (w_prev, askArray, bidArray, ExRet, sigmaSq, vcov, Lambda),
                                   method = 'trust-constr', 
                                   bounds = bounds,
                                   constraints = linear_constraint,
                                   options = {'disp': False})
        w_TC = w_TC.append(pd.Series(result.x, index=w_TC.columns), ignore_index=True)
        
        # update data for next period    
        data = data.append(pd.Series([np.log(1+validation['JCIExRet'].iloc[t]),
                                      np.log(1+validation['BondExRet'].iloc[t])], index=data.columns), 
                           ignore_index=True)
        w_prev = result.x

    return w_TC


# Function to compute portfolio weights based on Regime Switching model (2 regimes)
def regSw2(Lambda) : 
    #---------------------------------------------------------#
    # Asset allocation for regime switching model (2 regimes) #
    #---------------------------------------------------------#
    
    w_regSwitch2 = pd.DataFrame(columns=['JCI', 'Bond'])

    # use data in both train and validation set to estimate regime probabilities
    data = pd.DataFrame()
    data['JCILogRet'] = np.log(1+train['JCIExRet'])
    data['bondLogRet'] = np.log(1+train['BondExRet'])

    dataValidation = {'JCILogRet' : np.log(1+validation['JCIExRet']), 
                      'bondLogRet' : np.log(1+validation['BondExRet'])}
    data = data.append(pd.DataFrame(dataValidation))

    m = 2 # set number of regimes

    # import previously ran parameters
    theta22 = np.array(pd.read_csv("2var2regime_id_weekly_log.csv")['x'])

    # use the parameters to estimate filtered regime probabilities
    Filter = markovRegSwitch(theta22, data, m)[1]
    Filter['t index'] = list(range(0, len(data)))
    Filter['Regime 1'] = Filter['p1']>0.5
    Filter['Regime 2'] = Filter['p2']>0.5

    # construct probability transition matrix 
    p = np.matrix([[theta22[5], 1-theta22[5]], 
                   [theta22[11], 1-theta22[11]]])

    # mean and variance-covariance matrix for each regime
    nPar = int(len(theta22)/m)
    mu = [] 
    sig = []
    for i in range(0,m):
        mu.append(theta22[nPar*i:nPar*i+2])
        sig.append(np.matrix([[theta22[nPar*i+2], theta22[nPar*i+4]], 
                              [theta22[nPar*i+4], theta22[nPar*i+3]]]))

    # subset Filter to contain only lag data for the validation set 
    Filter = Filter.iloc[len(train)-1:len(train)+len(validation)-1,:]

    for t in range(0,len(validation)) :
        # start week 1 by constructing expected mean and var-cov matrix 
        # from probabilites of regime 1 and 2 and prior of prev week
        xi = np.matrix(Filter[['p1', 'p2']].iloc[t])
        xi_next=p.T*xi.T

        ExRet = np.matrix(mu).T*xi_next
        vcov = sig[0]*float(xi_next[0]) + sig[1]*float(xi_next[1])
        sigmaSq = np.diag(vcov)
    
        # set bounds for MVE optimisation (no shorting and total weights = 0.95)
        # set aside 5% of cash to pay transaction costs
        bounds = Bounds([0, 0], [0.95,0.95])
        linear_constraint = LinearConstraint([1, 1], [0], [0.95])
    
        # set initial weights
        w0=np.array([0.4, 0.4]) 
        # run constraint-bound optimisation to determine assets weight
        result = optimize.minimize(fun = utilLog, 
                                   x0 = w0, 
                                   args = (ExRet, sigmaSq, vcov, Lambda),
                                   method = 'trust-constr', 
                                   bounds = bounds,
                                   constraints = linear_constraint,
                                   options = {'disp': False})
        w_regSwitch2 = w_regSwitch2.append(pd.Series(result.x, 
                                                     index=w_regSwitch2.columns), 
                                           ignore_index=True)

    return w_regSwitch2
    

# Function to compute portfolio weights based on Regime Switching model (2 regimes)
# with Kalman Filter to smoothen asset expected returns to reduce fluctuations
# in asset weights
def regSw2Kalman(Lambda) : 
    #-------------------------------------------------------------------------#
    # Asset allocation for regime switching model (2 regimes) + Kalman Filter #
    #-------------------------------------------------------------------------#
    # Kalman filter algorithm is used to smoothen the expected returns and variances
    # At t=0, in-sample returns and variances are used as the prior 
    # Expected returns and variances from regime switching model are used as the 
    # 'new observation' to update the optimal estimate

    w_regSwitch2_Kalman = pd.DataFrame(columns=['JCI', 'Bond'])

    # use data in both train and validation set to estimate regime probabilities
    data = pd.DataFrame()
    data['JCILogRet'] = np.log(1+train['JCIExRet'])
    data['bondLogRet'] = np.log(1+train['BondExRet'])

    dataValidation = {'JCILogRet' : np.log(1+validation['JCIExRet']), 
                      'bondLogRet' : np.log(1+validation['BondExRet'])}
    data = data.append(pd.DataFrame(dataValidation))

    m = 2 # set number of regimes

    # import previously ran parameters
    theta22 = np.array(pd.read_csv("2var2regime_id_weekly_log.csv")['x'])

    # use the parameters to estimate filtered regime probabilities
    Filter = markovRegSwitch(theta22, data, m)[1]
    Filter['t index'] = list(range(0, len(data)))
    Filter['Regime 1'] = Filter['p1']>0.5
    Filter['Regime 2'] = Filter['p2']>0.5

    # construct probability transition matrix 
    p = np.matrix([[theta22[5], 1-theta22[5]], 
                   [theta22[11], 1-theta22[11]]])

    # mean and variance-covariance matrix for each regime
    nPar = int(len(theta22)/m)
    mu = [] 
    sig = []
    for i in range(0,m):
        mu.append(theta22[nPar*i:nPar*i+2])
        sig.append(np.matrix([[theta22[nPar*i+2], theta22[nPar*i+4]], 
                              [theta22[nPar*i+4], theta22[nPar*i+3]]]))

    # subset Filter to contain only lag data for the validation set 
    Filter = Filter.iloc[len(train)-1:len(train)+len(validation)-1,:]
    
    # Store Kalman filter parameters generated every week
    parKalman = pd.DataFrame(np.zeros((len(validation), 17)), 
                             columns=['priorJCI', 'priorBond', 
                                      'priorVarJCI', 'priorVarBond', 'priorCovar',
                                      'obsJCI', 'obsBond', 
                                      'obsVarJCI', 'obsVarBond', 'obsCovar',
                                      'postJCI', 'postBond', 
                                      'postVarJCI', 'postVarBond', 'postCovar',
                                      'gainJCI', 'gainBond'])

    # compute in-sample returns and variances for t=0 prior
    parKalman.loc[0, 'priorJCI'] = np.mean(train['JCIExRet'])
    parKalman.loc[0, 'priorBond'] = np.mean(train['BondExRet'])
    prior_vcov = np.cov([train['JCIExRet'], train['BondExRet']])
    parKalman.loc[0, 'priorVarJCI'], parKalman.loc[0, 'priorVarBond'] = np.diag(prior_vcov)
    parKalman.loc[0, 'priorCovar'] = prior_vcov[0,1]

    for t in range(0,len(validation)) :
        # Start week 1 by constructing expected mean and var-cov matrix 
        # from probabilites of regime 1 and 2 and prior of prev week
        xi = np.matrix(Filter[['p1', 'p2']].iloc[t])
        xi_next=p.T*xi.T
    
        # Construct vectors of observations and their variances (from regime switching)
        Y = np.matrix(mu).T*xi_next  
    
        R = sig[0]*float(xi_next[0]) + sig[1]*float(xi_next[1]) 
        
        # Construct vectors of priors and their variances
        X = np.array([[parKalman.loc[t, 'priorJCI']], 
                      [parKalman.loc[t, 'priorBond']]])
        P = np.array([[parKalman.loc[t, 'priorVarJCI'], parKalman.loc[t, 'priorCovar']],
                      [parKalman.loc[t, 'priorCovar'], parKalman.loc[t, 'priorVarBond']]])
    
        # Calculating the Kalman Gain
        H = np.identity(len(X))
        S = H.dot(P).dot(H.T) + R
        K = P.dot(H).dot(np.linalg.inv(S))

        # Update the State Matrix
        # Combination of the prior state, observed values, covariance matrix and Kalman Gain
        X_hat = X + K.dot(Y - H.dot(X))  

        # Update Process Covariance Matrix
        P_hat = (np.identity(len(K)) - K.dot(H)).dot(P)
    
        # store parameter values and update prior for next iteration 
        parKalman.loc[t, 'obsJCI'], parKalman.loc[t, 'obsBond'] = Y
        parKalman.loc[t, 'obsVarJCI'], parKalman.loc[t, 'obsVarBond'] = np.diag(R)
        parKalman.loc[t, 'obsCovar'] = R[0,1]
    
        parKalman.loc[t, 'gainJCI'], parKalman.loc[t, 'gainBond'] = np.diag(K)
    
        parKalman.loc[t, 'postJCI'], parKalman.loc[t, 'postBond'] = X_hat
        parKalman.loc[t, 'postVarJCI'], parKalman.loc[t, 'postVarBond'] = np.diag(P_hat)
        parKalman.loc[t, 'postCovar'] = P_hat[0,1]
    
        parKalman.loc[t+1, 'priorJCI'], parKalman.loc[t+1, 'priorBond'] = X_hat
        parKalman.loc[t+1, 'priorVarJCI'], parKalman.loc[t+1, 'priorVarBond'] = np.diag(P_hat)
        parKalman.loc[t+1, 'priorCovar'] = P_hat[0,1]
    
        # setup parameters for Mean-Variance Optimisation
        ExRet = X_hat
        vcov = P_hat
        sigmaSq = np.diag(vcov)
    
        # set bounds for MVE optimisation (no shorting and total weights = 0.95)
        # set aside 5% of cash to pay transaction costs
        bounds = Bounds([0, 0], [0.95,0.95])
        linear_constraint = LinearConstraint([1, 1], [0], [0.95])
    
        # set initial weights
        w0=np.array([0.4, 0.4]) 
        # run constraint-bound optimisation to determine assets weight
        result = optimize.minimize(fun = utilLog, 
                                   x0 = w0, 
                                   args = (ExRet, sigmaSq, vcov, Lambda),
                                   method = 'trust-constr', 
                                   bounds = bounds,
                                   constraints = linear_constraint,
                                   options = {'disp': False})
        w_regSwitch2_Kalman = w_regSwitch2_Kalman.append(pd.Series(result.x, 
                                                                   index=w_regSwitch2_Kalman.columns), 
                                                         ignore_index=True)
    
    return w_regSwitch2_Kalman


# Function to compute portfolio weights based on Regime Switching model (2 regimes) 
# and Black-Litterman with relative portfolio view
def regSw2BLrel(optimVar) :
    #---------------------------------------------------------------------------#
    # Asset allocation for regime switching model (2 regimes) + Black-Litterman #
    # with relative view on assets' expected returns                            #
    #---------------------------------------------------------------------------#
    # assets' expected returns from markov regime switching model are used to create 
    # view portfolio in Black-Litterman model. Equilibrium weights (prior weights) 
    # are computed from the market cap of both assets at the end of previous week  
    Lambda = optimVar[0]
    tau = optimVar[1]
    
    w_regSwitch2_BL = pd.DataFrame(columns=['JCI', 'Bond'])

    # use data in both train and validation set to estimate regime probabilities
    data = pd.DataFrame()
    data['JCILogRet'] = np.log(1+train['JCIExRet'])
    data['bondLogRet'] = np.log(1+train['BondExRet'])

    dataValidation = {'JCILogRet' : np.log(1+validation['JCIExRet']), 
                      'bondLogRet' : np.log(1+validation['BondExRet'])}
    data = data.append(pd.DataFrame(dataValidation))

    m = 2 # set number of regimes

    # import previously ran parameters
    theta22 = np.array(pd.read_csv("2var2regime_id_weekly_log.csv")['x'])
    
    # use the parameters to estimate filtered regime probabilities
    Filter = markovRegSwitch(theta22, data, m)[1]
    Filter['t index'] = list(range(0, len(data)))
    Filter['Regime 1'] = Filter['p1']>0.5
    Filter['Regime 2'] = Filter['p2']>0.5

    # construct probability transition matrix 
    p = np.matrix([[theta22[5], 1-theta22[5]], 
                   [theta22[11], 1-theta22[11]]])

    # mean and variance-covariance matrix for each regime
    nPar = int(len(theta22)/m)
    mu = [] 
    sig = []
    for i in range(0,m):
        mu.append(theta22[nPar*i:nPar*i+2])
        sig.append(np.matrix([[theta22[nPar*i+2], theta22[nPar*i+4]], 
                              [theta22[nPar*i+4], theta22[nPar*i+3]]]))

    # subset Filter to contain only lag data for the validation set 
    Filter = Filter.iloc[len(train)-1:len(train)+len(validation)-1,:]
    
    # subset data to only include rows with available market cap data
    data = train.dropna()

    # Store parameters generated every week
    parBL = pd.DataFrame(np.zeros((len(validation), 10)), 
                                  columns=['Lambda', 'Pi_JCI', 'Pi_Bond', 'RegSwJCI', 
                                           'RegSwBond', 'omega', 'ExRetJCI', 'ExRetBond',
                                           'varJCI', 'varBond'])

    for t in range(0,len(validation)) :
        # compute Black-Litterman model parameters 
        # risk aversion coefficient Lambda_BL
        # lambda = (expected excess log-return + 0.5*variance of log return) / 
        # variance of log return of the equilibrium portfolio
        #Lambda_BL = (np.mean(np.log(1 + data['mktCapPortExRet'])) +\
        #             0.5*np.var(np.log(1+data['mktCapPortRet'])))/np.var(np.log(1+data['mktCapPortRet']))
        Lambda_BL = Lambda
        #Lambda_BL = np.mean(np.log(1 + data.iloc[len(data)-103:len(data),:]['mktCapPortExRet']))/\
        #    np.var(np.log(1+data.iloc[len(data)-103:len(data),:]['mktCapPortRet']))
        parBL.loc[t,'Lambda'] = Lambda_BL

        # Sigma (variance covariance matrix of assets' log returns)
        sig = np.cov([np.log(1+data['JCIWeeklyRet']), np.log(1+data['BondWeeklyRet'])])
        #sig = np.cov([np.log(1+data.iloc[len(data)-103:len(data),:]['JCIWeeklyRet']), 
        #              np.log(1+data.iloc[len(data)-103:len(data),:]['BondWeeklyRet'])])
        
        # equilibrium weights w_eq from the markep cap weighted portfolio
        w_eq = np.array([data.iloc[len(data)-1,:]['w'], 1-data.iloc[len(data)-1,:]['w']])
    
        # implied equilibrium return Pi
        Pi = Lambda_BL*np.matmul(sig, w_eq)
        parBL.loc[t,'Pi_JCI'] = Pi[0]
        parBL.loc[t,'Pi_Bond'] = Pi[1]
        
        # create view portfolio (Q and P) from regime switching model
        # start week 1 by constructing assets' expected return from probability of regime 1 and 2 and prior of prev week
        xi = np.matrix(Filter[['p1', 'p2']].iloc[t])
        xi_next=p.T*xi.T
        ExRet = np.matrix(mu).T*xi_next
        parBL.loc[t,'RegSwJCI'] = ExRet[0,0]
        parBL.loc[t,'RegSwBond'] = ExRet[1,0]

        # construct view portfolio, with view on relative performance    
        if(ExRet[0,0] >= ExRet[1,0]) : 
            Q = ExRet[0,0] - ExRet[1,0]
            P = np.matrix([1, -1])
        elif(ExRet[0,0] < ExRet[1,0]) : 
            Q = ExRet[1,0] - ExRet[0,0]
            P = np.matrix([-1, 1])
        
        # compute uncertainty of view (omega = tau*P*Sigma*P')
        omega = tau*np.matmul(np.matmul(P, sig), P.T)
        parBL.loc[t,'omega'] = omega
    
        # compute assets' expected return and variance-covariance matrix using Black-Litterman equation
        vcov = np.linalg.inv(np.linalg.inv(tau*sig) + np.matmul(np.matmul(P.T, np.linalg.inv(omega)), P))
        parBL.loc[t,'varJCI'] = vcov[0,0]
        parBL.loc[t,'varBond'] = vcov[1,1]
    
        ExRet = np.matmul(vcov, (np.matmul(np.linalg.inv(tau*sig), Pi).reshape(2,1) +\
                                 np.matmul(P.T, np.linalg.inv(omega))*Q))
        parBL.loc[t,'ExRetJCI'] = ExRet[0,0]
        parBL.loc[t,'ExRetBond'] = ExRet[1,0]

        sigmaSq = np.diag(vcov)
    
        # set bounds for MVE optimisation (no shorting and total weights = 0.95)
        # set aside 5% of cash to pay transaction costs
        bounds = Bounds([0, 0], [0.95,0.95])
        linear_constraint = LinearConstraint([1, 1], [0], [0.95])
    
        # set initial weights
        w0=np.array([0.4, 0.4]) 
        # run constraint-bound optimisation to determine assets weight
        result = optimize.minimize(fun = utilLog, 
                                   x0 = w0, 
                                   args =(ExRet, sigmaSq, vcov, Lambda),
                                   method = 'trust-constr', 
                                   bounds = bounds,
                                   constraints = linear_constraint,
                                   options = {'disp': False})
        w_regSwitch2_BL = w_regSwitch2_BL.append(pd.Series(result.x, 
                                                           index=w_regSwitch2_BL.columns),
                                                 ignore_index=True)
    
        # update data for next period    
        data = data.append(validation.iloc[t,:])

    return w_regSwitch2_BL


# Function to compute portfolio weights based on Regime Switching model (2 regimes) 
# and Black-Litterman with absolute portfolio view
def regSw2BLabs(optimVar) :
    #---------------------------------------------------------------------------#
    # Asset allocation for regime switching model (2 regimes) + Black-Litterman #
    # with absolute view on assets' expected returns                            #
    #---------------------------------------------------------------------------#
    # assets' expected returns from markov regime switching model are used to create 
    # view portfolio in Black-Litterman model. Equilibrium weights (prior weights) 
    # are computed from the market cap of both assets at the end of previous week 
    Lambda = optimVar[0]
    tau = optimVar[1]
    
    w_regSwitch2_BL = pd.DataFrame(columns=['JCI', 'Bond'])

    # use data in both train and validation set to estimate regime probabilities
    data = pd.DataFrame()
    data['JCILogRet'] = np.log(1+train['JCIExRet'])
    data['bondLogRet'] = np.log(1+train['BondExRet'])

    dataValidation = {'JCILogRet' : np.log(1+validation['JCIExRet']), 
                      'bondLogRet' : np.log(1+validation['BondExRet'])}
    data = data.append(pd.DataFrame(dataValidation))

    m = 2 # set number of regimes

    # import previously ran parameters
    theta22 = np.array(pd.read_csv("2var2regime_id_weekly_log.csv")['x'])

    # use the parameters to estimate filtered regime probabilities
    Filter = markovRegSwitch(theta22, data, m)[1]
    Filter['t index'] = list(range(0, len(data)))
    Filter['Regime 1'] = Filter['p1']>0.5
    Filter['Regime 2'] = Filter['p2']>0.5

    # construct probability transition matrix 
    p = np.matrix([[theta22[5], 1-theta22[5]], 
                   [theta22[11], 1-theta22[11]]])

    # mean and variance-covariance matrix for each regime
    nPar = int(len(theta22)/m)
    mu = [] 
    sig = []
    for i in range(0,m):
        mu.append(theta22[nPar*i:nPar*i+2])
        sig.append(np.matrix([[theta22[nPar*i+2], theta22[nPar*i+4]], 
                              [theta22[nPar*i+4], theta22[nPar*i+3]]]))

    # subset Filter to contain only lag data for the validation set 
    Filter = Filter.iloc[len(train)-1:len(train)+len(validation)-1,:]

    # subset data to only include rows with available market cap data
    data = train.dropna()

    # Store parameters generated every week
    parBL = pd.DataFrame(np.zeros((len(validation), 11)), 
                                  columns=['Lambda', 'Pi_JCI', 'Pi_Bond', 'RegSwJCI', 
                                           'RegSwBond', 'omegaJCI', 'omegaBond', 'ExRetJCI', 'ExRetBond',
                                           'varJCI', 'varBond'])

    for t in range(0,len(validation)) :
        # compute Black-Litterman model parameters 
        # risk aversion coefficient Lambda_BL
        # lambda = (expected excess log-return + 0.5*variance of log return) / 
        # variance of log return of the equilibrium portfolio
        #Lambda_BL = (np.mean(np.log(1 + data['mktCapPortExRet'])) +\
        #             0.5*np.var(np.log(1+data['mktCapPortRet'])))/np.var(np.log(1+data['mktCapPortRet']))
        Lambda_BL = Lambda 
        #Lambda_BL = np.mean(np.log(1 + data.iloc[len(data)-103:len(data),:]['mktCapPortExRet']))/\
        #    np.var(np.log(1+data.iloc[len(data)-103:len(data),:]['mktCapPortRet']))
        parBL.loc[t,'Lambda'] = Lambda_BL
        
        # Sigma (variance covariance matrix of assets' log returns)
        sig = np.cov([np.log(1+data['JCIWeeklyRet']), np.log(1+data['BondWeeklyRet'])])
        #sig = np.cov([np.log(1+data.iloc[len(data)-103:len(data),:]['JCIWeeklyRet']), 
        #              np.log(1+data.iloc[len(data)-103:len(data),:]['BondWeeklyRet'])])
        
        # equilibrium weights w_eq from the markep cap weighted portfolio
        w_eq = np.array([data.iloc[len(data)-1,:]['w'], 1-data.iloc[len(data)-1,:]['w']])
    
        # implied equilibrium return Pi
        Pi = Lambda_BL*np.matmul(sig, w_eq)
        parBL.loc[t,'Pi_JCI'] = Pi[0]
        parBL.loc[t,'Pi_Bond'] = Pi[1]
    
        # create view portfolio (Q and P) from regime switching model
        # start week 1 by constructing assets' expected return from probability of regime 1 and 2 and prior of prev week
        xi = np.matrix(Filter[['p1', 'p2']].iloc[t])
        xi_next=p.T*xi.T
        ExRet = np.matrix(mu).T*xi_next
        parBL.loc[t,'RegSwJCI'] = ExRet[0,0]
        parBL.loc[t,'RegSwBond'] = ExRet[1,0]
            
        # construct absolute view portfolio based on the regime switching model expected return
        Q = ExRet
        P = np.matrix([[1, 0], [0,1]])
        
        # compute uncertainty of views using variance from regime switching
        vcov_regSwitch = sig[0]*float(xi_next[0]) + sig[1]*float(xi_next[1])
        omega = np.diag(vcov_regSwitch)
        #omega = tau*np.matmul(np.matmul(P, sig), P.T)
        parBL.loc[t,'omegaJCI'] = omega[0,0]
        parBL.loc[t,'omegaBond'] = omega[1,1]
        
        # compute assets' expected return and variance-covariance matrix using Black-Litterman equation
        vcov = np.linalg.inv(np.linalg.inv(tau*sig) + np.matmul(np.matmul(P.T, np.linalg.inv(omega)), P))
        parBL.loc[t,'varJCI'] = vcov[0,0]
        parBL.loc[t,'varBond'] = vcov[1,1]
    
        ExRet = np.matmul(vcov, (np.matmul(np.linalg.inv(tau*sig), Pi).reshape(2,1) +\
                                 np.matmul(P.T, np.linalg.inv(omega))*Q))
        parBL.loc[t,'ExRetJCI'] = ExRet[0,0]
        parBL.loc[t,'ExRetBond'] = ExRet[1,0]

        sigmaSq = np.diag(vcov)
    
        # set bounds for MVE optimisation (no shorting and total weights = 0.95)
        # set aside 5% of cash to pay transaction costs
        bounds = Bounds([0, 0], [0.95,0.95])
        linear_constraint = LinearConstraint([1, 1], [0], [0.95])
    
        # set initial weights
        w0=np.array([0.4, 0.4]) 
        # run constraint-bound optimisation to determine assets weight
        # Lambda = Lambda_BL # set Lambda in MVE equals to Lambda implied by equilibrium portfolio
        result = optimize.minimize(fun = utilLog, 
                                   x0 = w0, 
                                   args = (ExRet, sigmaSq, vcov, Lambda),
                                   method = 'trust-constr', 
                                   bounds = bounds,
                                   constraints = linear_constraint,
                                   options = {'disp': False})
        w_regSwitch2_BL = w_regSwitch2_BL.append(pd.Series(result.x, 
                                                           index=w_regSwitch2_BL.columns),
                                                 ignore_index=True)
    
        # update data for next period    
        data = data.append(validation.iloc[t,:])

    return w_regSwitch2_BL


# Function to compute portfolio weights based on Regime Switching model (2 regimes)
# with Transaction Cost in the MVO algorithm
def regSw2_TC(Lambda) : 
    #---------------------------------------------------------#
    # Asset allocation for regime switching model (2 regimes) #
    # including transaction costs                             #
    #---------------------------------------------------------#
    
    w_regSwitch2_TC = pd.DataFrame(columns=['JCI', 'Bond'])

    # use data in both train and validation set to estimate regime probabilities
    data = pd.DataFrame()
    data['JCILogRet'] = np.log(1+train['JCIExRet'])
    data['bondLogRet'] = np.log(1+train['BondExRet'])

    dataValidation = {'JCILogRet' : np.log(1+validation['JCIExRet']), 
                      'bondLogRet' : np.log(1+validation['BondExRet'])}
    data = data.append(pd.DataFrame(dataValidation))
    
    m = 2 # set number of regimes

    # import previously ran parameters
    theta22 = np.array(pd.read_csv("2var2regime_id_weekly_log.csv")['x'])

    # use the parameters to estimate filtered regime probabilities
    Filter = markovRegSwitch(theta22, data, m)[1]
    Filter['t index'] = list(range(0, len(data)))
    Filter['Regime 1'] = Filter['p1']>0.5
    Filter['Regime 2'] = Filter['p2']>0.5
    
    # construct probability transition matrix 
    p = np.matrix([[theta22[5], 1-theta22[5]], 
                   [theta22[11], 1-theta22[11]]])
    
    # mean and variance-covariance matrix for each regime
    nPar = int(len(theta22)/m)
    mu = [] 
    sig = []
    for i in range(0,m):
        mu.append(theta22[nPar*i:nPar*i+2])
        sig.append(np.matrix([[theta22[nPar*i+2], theta22[nPar*i+4]], 
                              [theta22[nPar*i+4], theta22[nPar*i+3]]]))

    # subset Filter to contain only lag data for the validation set 
    Filter = Filter.iloc[len(train)-1:len(train)+len(validation)-1,:]
    
    # set weights at t=0 to be 0
    w_prev = np.array([0, 0]) 
    # array of bid and ask costs, assuming same unit costs for all assets
    askArray = np.array([ask, ask])
    bidArray = np.array([bid, bid])

    for t in range(0,len(validation)) :
        # start week 1 by constructing expected mean and var-cov matrix 
        # from probabilites of regime 1 and 2 and prior of prev week
        xi = np.matrix(Filter[['p1', 'p2']].iloc[t])
        xi_next=p.T*xi.T

        ExRet = np.matrix(mu).T*xi_next
        vcov = sig[0]*float(xi_next[0]) + sig[1]*float(xi_next[1])
        sigmaSq = np.diag(vcov)
    
        # set bounds for MVE optimisation (no shorting and total weights = 0.95)
        # set aside 5% of cash to pay transaction costs
        bounds = Bounds([0, 0], [0.95,0.95])
        linear_constraint = LinearConstraint([1, 1], [0], [0.95])
    
        # set initial weights
        w0=np.array([0.4, 0.4]) 
        # run constraint-bound optimisation to determine assets weight
        result = optimize.minimize(fun = utilLogTC, 
                                   x0 = w0, 
                                   args = (w_prev, askArray, bidArray, ExRet, sigmaSq, vcov, Lambda),
                                   method = 'trust-constr', 
                                   bounds = bounds,
                                   constraints = linear_constraint,
                                   options = {'disp': False})
        w_regSwitch2_TC = w_regSwitch2_TC.append(pd.Series(result.x, 
                                                           index=w_regSwitch2_TC.columns), 
                                                 ignore_index=True)
    
        # update data for next period
        w_prev = result.x
        
    return w_regSwitch2_TC


# Function to compute portfolio weights based on Regime Switching model (2 regimes) 
# and Black-Litterman with absolute portfolio view
# with Transaction Cost in the MVO algorithm
def regSw2BLabs_TC(optimVar) :
    #---------------------------------------------------------------------------#
    # Asset allocation for regime switching model (2 regimes) + Black-Litterman #
    # with absolute view on assets' expected returns and                        #
    # including transaction costs                                               #
    #---------------------------------------------------------------------------#
    # assets' expected returns from markov regime switching model are used to create 
    # view portfolio in Black-Litterman model. Equilibrium weights (prior weights) 
    # are computed from the market cap of both assets at the end of previous week 
    Lambda = optimVar[0]
    tau = optimVar[1]
    
    w_regSwitch2_BL_TC = pd.DataFrame(columns=['JCI', 'Bond'])

    # use data in both train and validation set to estimate regime probabilities
    data = pd.DataFrame()
    data['JCILogRet'] = np.log(1+train['JCIExRet'])
    data['bondLogRet'] = np.log(1+train['BondExRet'])

    dataValidation = {'JCILogRet' : np.log(1+validation['JCIExRet']), 
                      'bondLogRet' : np.log(1+validation['BondExRet'])}
    data = data.append(pd.DataFrame(dataValidation))

    m = 2 # set number of regimes

    # import previously ran parameters
    theta22 = np.array(pd.read_csv("2var2regime_id_weekly_log.csv")['x'])

    # use the parameters to estimate filtered regime probabilities
    Filter = markovRegSwitch(theta22, data, m)[1]
    Filter['t index'] = list(range(0, len(data)))
    Filter['Regime 1'] = Filter['p1']>0.5
    Filter['Regime 2'] = Filter['p2']>0.5

    # construct probability transition matrix 
    p = np.matrix([[theta22[5], 1-theta22[5]], 
                   [theta22[11], 1-theta22[11]]])

    # mean and variance-covariance matrix for each regime
    nPar = int(len(theta22)/m)
    mu = [] 
    sig = []
    for i in range(0,m):
        mu.append(theta22[nPar*i:nPar*i+2])
        sig.append(np.matrix([[theta22[nPar*i+2], theta22[nPar*i+4]], 
                              [theta22[nPar*i+4], theta22[nPar*i+3]]]))

    # subset Filter to contain only lag data for the validation set 
    Filter = Filter.iloc[len(train)-1:len(train)+len(validation)-1,:]

    # subset data to only include rows with available market cap data
    data = train.dropna()

    # Store parameters generated every week
    parBL = pd.DataFrame(np.zeros((len(validation), 11)), 
                                  columns=['Lambda', 'Pi_JCI', 'Pi_Bond', 'RegSwJCI', 
                                           'RegSwBond', 'omegaJCI', 'omegaBond', 'ExRetJCI', 'ExRetBond',
                                           'varJCI', 'varBond'])
    
    # set weights at t=0 to be 0
    w_prev = np.array([0, 0]) 
    # array of bid and ask costs, assuming same unit costs for all assets
    askArray = np.array([ask, ask])
    bidArray = np.array([bid, bid])

    for t in range(0,len(validation)) :
        # compute Black-Litterman model parameters 
        # risk aversion coefficient Lambda_BL
        # lambda = (expected excess log-return + 0.5*variance of log return) / 
        # variance of log return of the equilibrium portfolio
        #Lambda_BL = (np.mean(np.log(1 + data['mktCapPortExRet'])) +\
        #             0.5*np.var(np.log(1+data['mktCapPortRet'])))/np.var(np.log(1+data['mktCapPortRet']))
        Lambda_BL = Lambda 
        #Lambda_BL = np.mean(np.log(1 + data.iloc[len(data)-103:len(data),:]['mktCapPortExRet']))/\
        #    np.var(np.log(1+data.iloc[len(data)-103:len(data),:]['mktCapPortRet']))
        parBL.loc[t,'Lambda'] = Lambda_BL
        
        # Sigma (variance covariance matrix of assets' log returns)
        sig = np.cov([np.log(1+data['JCIWeeklyRet']), np.log(1+data['BondWeeklyRet'])])
        #sig = np.cov([np.log(1+data.iloc[len(data)-103:len(data),:]['JCIWeeklyRet']), 
        #              np.log(1+data.iloc[len(data)-103:len(data),:]['BondWeeklyRet'])])
        
        # equilibrium weights w_eq from the markep cap weighted portfolio
        w_eq = np.array([data.iloc[len(data)-1,:]['w'], 1-data.iloc[len(data)-1,:]['w']])
    
        # implied equilibrium return Pi
        Pi = Lambda_BL*np.matmul(sig, w_eq)
        parBL.loc[t,'Pi_JCI'] = Pi[0]
        parBL.loc[t,'Pi_Bond'] = Pi[1]
    
        # create view portfolio (Q and P) from regime switching model
        # start week 1 by constructing assets' expected return from probability of regime 1 and 2 and prior of prev week
        xi = np.matrix(Filter[['p1', 'p2']].iloc[t])
        xi_next=p.T*xi.T
        ExRet = np.matrix(mu).T*xi_next
        parBL.loc[t,'RegSwJCI'] = ExRet[0,0]
        parBL.loc[t,'RegSwBond'] = ExRet[1,0]
            
        # construct absolute view portfolio based on the regime switching model expected return
        Q = ExRet
        P = np.matrix([[1, 0], [0,1]])
        
        # compute uncertainty of views using variance from regime switching
        vcov_regSwitch = sig[0]*float(xi_next[0]) + sig[1]*float(xi_next[1])
        omega = np.diag(vcov_regSwitch)
        #omega = tau*np.matmul(np.matmul(P, sig), P.T)
        parBL.loc[t,'omegaJCI'] = omega[0,0]
        parBL.loc[t,'omegaBond'] = omega[1,1]
        
        # compute assets' expected return and variance-covariance matrix using Black-Litterman equation
        vcov = np.linalg.inv(np.linalg.inv(tau*sig) + np.matmul(np.matmul(P.T, np.linalg.inv(omega)), P))
        parBL.loc[t,'varJCI'] = vcov[0,0]
        parBL.loc[t,'varBond'] = vcov[1,1]
    
        ExRet = np.matmul(vcov, (np.matmul(np.linalg.inv(tau*sig), Pi).reshape(2,1) +\
                                 np.matmul(P.T, np.linalg.inv(omega))*Q))
        parBL.loc[t,'ExRetJCI'] = ExRet[0,0]
        parBL.loc[t,'ExRetBond'] = ExRet[1,0]

        sigmaSq = np.diag(vcov)
    
        # set bounds for MVE optimisation (no shorting and total weights = 0.95)
        # set aside 5% of cash to pay transaction costs
        bounds = Bounds([0, 0], [0.95,0.95])
        linear_constraint = LinearConstraint([1, 1], [0], [0.95])
    
        # set initial weights
        w0=np.array([0.4, 0.4]) 
        # run constraint-bound optimisation to determine assets weight
        # Lambda = Lambda_BL # set Lambda in MVE equals to Lambda implied by equilibrium portfolio
        result = optimize.minimize(fun = utilLogTC, 
                                   x0 = w0, 
                                   args = (w_prev, askArray, bidArray, ExRet, sigmaSq, vcov, Lambda),
                                   method = 'trust-constr', 
                                   bounds = bounds,
                                   constraints = linear_constraint,
                                   options = {'disp': False})
        w_regSwitch2_BL_TC = w_regSwitch2_BL_TC.append(pd.Series(result.x, 
                                                           index=w_regSwitch2_BL_TC.columns),
                                                 ignore_index=True)
    
        # update data for next period    
        data = data.append(validation.iloc[t,:])
        w_prev = result.x

    return w_regSwitch2_BL_TC


# Function to compute portfolio weights based on Regime Switching model (2 regimes)
def regSw4(Lambda) : 
    #---------------------------------------------------------#
    # Asset allocation for regime switching model (4 regimes) #
    #---------------------------------------------------------#

    w_regSwitch4 = pd.DataFrame(columns=['JCI', 'Bond'])

    # use data in both train and validation set to estimate regime probabilities
    data = pd.DataFrame()
    data['JCILogRet'] = np.log(1+train['JCIExRet'])
    data['bondLogRet'] = np.log(1+train['BondExRet'])

    dataValidation = {'JCILogRet' : np.log(1+validation['JCIExRet']), 
                      'bondLogRet' : np.log(1+validation['BondExRet'])}
    data = data.append(pd.DataFrame(dataValidation))

    m = 4 # set number of regimes

    # import previously ran parameters
    theta24 = np.array(pd.read_csv("2var4regime_id_weekly_log.csv")['x'])

    # use the parameters to estimate filtered regime probabilities
    Filter = markovRegSwitch(theta24, data, m)[1]
    Filter['t index'] = list(range(0, len(data)))
    Filter['Regime 1'] = Filter['p1']>0.5
    Filter['Regime 2'] = Filter['p2']>0.5
    Filter['Regime 3'] = Filter['p3']>0.5
    Filter['Regime 4'] = Filter['p4']>0.5

    # construct probability transition matrix 
    nPar = int(len(theta24)/m)
    p = np.zeros((m,m)) 
    for i in range(0,m):
        for j in range(0,m-1):
            p[i][j]=theta24[nPar*i+5+j]
        p[i][(m-1)]=1-sum(p[i][0:(m-1)])

    # mean and variance-covariance matrix for each regime
    mu = [] 
    sig = []
    for i in range(0,m):
        mu.append(theta24[nPar*i:nPar*i+2]) 
        sig.append(np.matrix([[theta24[nPar*i+2], theta24[nPar*i+4]], 
                              [theta24[nPar*i+4], theta24[nPar*i+3]]]))

    # subset Filter to contain only lag data for the validation set 
    Filter = Filter.iloc[len(train)-1:len(train)+len(validation)-1,:]

    for t in range(0,len(validation)) :
        # start week 1 by constructing expected mean and var-cov matrix 
        # from probabilites of regime 1 and 2 and prior of prev week
        xi = np.matrix(Filter[['p1', 'p2', 'p3', 'p4']].iloc[t])
        xi_next=p.T*xi.T

        ExRet = np.matrix(mu).T*xi_next
        vcov = sig[0]*float(xi_next[0]) + sig[1]*float(xi_next[1]) + \
                sig[2]*float(xi_next[2]) + sig[3]*float(xi_next[3])
        sigmaSq = np.diag(vcov)
    
        # set bounds for MVE optimisation (no shorting and total weights = 0.95)
        # set aside 5% of cash to pay transaction costs
        bounds = Bounds([0, 0], [0.95,0.95])
        linear_constraint = LinearConstraint([1, 1], [0], [0.95])
    
        # set initial weights
        w0=np.array([0.4, 0.4]) 
        # run constraint-bound optimisation to determine assets weight
        result = optimize.minimize(fun = utilLog, 
                                   x0 = w0, 
                                   args = (ExRet, sigmaSq, vcov, Lambda),
                                   method = 'trust-constr', 
                                   bounds = bounds,
                                   constraints = linear_constraint,
                                   options = {'disp': False})
        w_regSwitch4 = w_regSwitch4.append(pd.Series(result.x, 
                                                     index=w_regSwitch4.columns), 
                                           ignore_index=True)

    return w_regSwitch4
    
    
# Function to compute portfolio weights based on Regime Switching model (4 regimes)
# with Kalman Filter to smoothen asset expected returns to reduce fluctuations
# in asset weights
def regSw4Kalman(Lambda) : 
    #-------------------------------------------------------------------------#
    # Asset allocation for regime switching model (4 regimes) + Kalman Filter #
    #-------------------------------------------------------------------------#
    # Kalman filter algorithm is used to smoothen the expected returns and variances
    # At t=0, in-sample returns and variances are used as the prior 
    # Expected returns and variances from regime switching model are used as the 
    # 'new observation' to update the optimal estimate

    w_regSwitch4_Kalman = pd.DataFrame(columns=['JCI', 'Bond'])

    # use data in both train and validation set to estimate regime probabilities
    data = pd.DataFrame()
    data['JCILogRet'] = np.log(1+train['JCIExRet'])
    data['bondLogRet'] = np.log(1+train['BondExRet'])

    dataValidation = {'JCILogRet' : np.log(1+validation['JCIExRet']), 
                      'bondLogRet' : np.log(1+validation['BondExRet'])}
    data = data.append(pd.DataFrame(dataValidation))

    m = 4 # set number of regimes
    
    # import previously ran parameters
    theta24 = np.array(pd.read_csv("2var4regime_id_weekly_log.csv")['x'])

    # use the parameters to estimate filtered regime probabilities
    Filter = markovRegSwitch(theta24, data, m)[1]
    Filter['t index'] = list(range(0, len(data)))
    Filter['Regime 1'] = Filter['p1']>0.5
    Filter['Regime 2'] = Filter['p2']>0.5
    Filter['Regime 3'] = Filter['p3']>0.5
    Filter['Regime 4'] = Filter['p4']>0.5

    # construct probability transition matrix 
    nPar = int(len(theta24)/m)
    p = np.zeros((m,m)) 
    for i in range(0,m):
        for j in range(0,m-1):
            p[i][j]=theta24[nPar*i+5+j]
        p[i][(m-1)]=1-sum(p[i][0:(m-1)])

    # mean and variance-covariance matrix for each regime
    mu = [] 
    sig = []
    for i in range(0,m):
        mu.append(theta24[nPar*i:nPar*i+2]) 
        sig.append(np.matrix([[theta24[nPar*i+2], theta24[nPar*i+4]], 
                              [theta24[nPar*i+4], theta24[nPar*i+3]]]))

    # subset Filter to contain only lag data for the validation set 
    Filter = Filter.iloc[len(train)-1:len(train)+len(validation)-1,:]
    
    # Store Kalman filter parameters generated every week
    parKalman = pd.DataFrame(np.zeros((len(validation), 17)), 
                             columns=['priorJCI', 'priorBond', 
                                      'priorVarJCI', 'priorVarBond', 'priorCovar',
                                      'obsJCI', 'obsBond', 
                                      'obsVarJCI', 'obsVarBond', 'obsCovar',
                                      'postJCI', 'postBond', 
                                      'postVarJCI', 'postVarBond', 'postCovar',
                                      'gainJCI', 'gainBond'])

    # compute in-sample returns and variances for t=0 prior
    parKalman.loc[0, 'priorJCI'] = np.mean(train['JCIExRet'])
    parKalman.loc[0, 'priorBond'] = np.mean(train['BondExRet'])
    prior_vcov = np.cov([train['JCIExRet'], train['BondExRet']])
    parKalman.loc[0, 'priorVarJCI'], parKalman.loc[0, 'priorVarBond'] = np.diag(prior_vcov)
    parKalman.loc[0, 'priorCovar'] = prior_vcov[0,1]
    
    for t in range(0,len(validation)) :
        # start week 1 by constructing expected mean and var-cov matrix 
        # from probabilites of regime 1 and 2 and prior of prev week
        xi = np.matrix(Filter[['p1', 'p2', 'p3', 'p4']].iloc[t])
        xi_next=p.T*xi.T
    
        # Construct vectors of observations and their variances (from regime switching)
        Y = np.matrix(mu).T*xi_next  
        
        R = sig[0]*float(xi_next[0]) + sig[1]*float(xi_next[1]) + \
            sig[2]*float(xi_next[2]) + sig[3]*float(xi_next[3])

        # Construct vectors of priors and their variances
        X = np.array([[parKalman.loc[t, 'priorJCI']], 
                      [parKalman.loc[t, 'priorBond']]])
        P = np.array([[parKalman.loc[t, 'priorVarJCI'], parKalman.loc[t, 'priorCovar']],
                      [parKalman.loc[t, 'priorCovar'], parKalman.loc[t, 'priorVarBond']]])
    
        # Calculating the Kalman Gain
        H = np.identity(len(X))
        S = H.dot(P).dot(H.T) + R
        K = P.dot(H).dot(np.linalg.inv(S))

        # Update the State Matrix
        # Combination of the prior state, observed values, covariance matrix and Kalman Gain
        X_hat = X + K.dot(Y - H.dot(X))  

        # Update Process Covariance Matrix
        P_hat = (np.identity(len(K)) - K.dot(H)).dot(P)
    
        # store parameter values and update prior for next iteration 
        parKalman.loc[t, 'obsJCI'], parKalman.loc[t, 'obsBond'] = Y
        parKalman.loc[t, 'obsVarJCI'], parKalman.loc[t, 'obsVarBond'] = np.diag(R)
        parKalman.loc[t, 'obsCovar'] = R[0,1]
    
        parKalman.loc[t, 'gainJCI'], parKalman.loc[t, 'gainBond'] = np.diag(K)
    
        parKalman.loc[t, 'postJCI'], parKalman.loc[t, 'postBond'] = X_hat
        parKalman.loc[t, 'postVarJCI'], parKalman.loc[t, 'postVarBond'] = np.diag(P_hat)
        parKalman.loc[t, 'postCovar'] = P_hat[0,1]
    
        parKalman.loc[t+1, 'priorJCI'], parKalman.loc[t+1, 'priorBond'] = X_hat
        parKalman.loc[t+1, 'priorVarJCI'], parKalman.loc[t+1, 'priorVarBond'] = np.diag(P_hat)
        parKalman.loc[t+1, 'priorCovar'] = P_hat[0,1]
    
        # setup parameters for Mean-Variance Optimisation
        ExRet = X_hat
        vcov = P_hat
        sigmaSq = np.diag(vcov)
    
        # set bounds for MVE optimisation (no shorting and total weights = 0.95)
        # set aside 5% of cash to pay transaction costs
        bounds = Bounds([0, 0], [0.95,0.95])
        linear_constraint = LinearConstraint([1, 1], [0], [0.95])
    
        # set initial weights
        w0=np.array([0.4, 0.4]) 
        # run constraint-bound optimisation to determine assets weight
        result = optimize.minimize(fun = utilLog, 
                                   x0 = w0, 
                                   args = (ExRet, sigmaSq, vcov, Lambda),
                                   method = 'trust-constr', 
                                   bounds = bounds,
                                   constraints = linear_constraint,
                                   options = {'disp': False})
        w_regSwitch4_Kalman = w_regSwitch4_Kalman.append(pd.Series(result.x, 
                                                                   index=w_regSwitch4_Kalman.columns), 
                                                         ignore_index=True)
        
    return w_regSwitch4_Kalman


# Function to compute portfolio weights based on Regime Switching model (4 regimes) 
# and Black-Litterman with relative portfolio view
def regSw4BLrel(optimVar) :
    #---------------------------------------------------------------------------#
    # Asset allocation for regime switching model (4 regimes) + Black-Litterman #
    # with relative view on assets' expected returns                            #
    #---------------------------------------------------------------------------#
    # assets' expected returns from markov regime switching model are used to create 
    # view portfolio in Black-Litterman model. Equilibrium weights (prior weights) 
    # are computed from the market cap of both assets at the end of previous week  
    Lambda = optimVar[0]
    tau = optimVar[1]

    w_regSwitch4_BL = pd.DataFrame(columns=['JCI', 'Bond'])

    # use data in both train and validation set to estimate regime probabilities
    data = pd.DataFrame()
    data['JCILogRet'] = np.log(1+train['JCIExRet'])
    data['bondLogRet'] = np.log(1+train['BondExRet'])

    dataValidation = {'JCILogRet' : np.log(1+validation['JCIExRet']), 
                      'bondLogRet' : np.log(1+validation['BondExRet'])}
    data = data.append(pd.DataFrame(dataValidation))

    m = 4 # set number of regimes

    # import previously ran parameters
    theta24 = np.array(pd.read_csv("2var4regime_id_weekly_log.csv")['x'])
    
    # use the parameters to estimate filtered regime probabilities
    Filter = markovRegSwitch(theta24, data, m)[1]
    Filter['t index'] = list(range(0, len(data)))
    Filter['Regime 1'] = Filter['p1']>0.5
    Filter['Regime 2'] = Filter['p2']>0.5
    Filter['Regime 3'] = Filter['p3']>0.5
    Filter['Regime 4'] = Filter['p4']>0.5

    # construct probability transition matrix 
    nPar = int(len(theta24)/m)
    p = np.zeros((m,m)) 
    for i in range(0,m):
        for j in range(0,m-1):
            p[i][j]=theta24[nPar*i+5+j]
        p[i][(m-1)]=1-sum(p[i][0:(m-1)])

    # mean and variance-covariance matrix for each regime
    mu = [] 
    sig = []
    for i in range(0,m):
        mu.append(theta24[nPar*i:nPar*i+2])
        sig.append(np.matrix([[theta24[nPar*i+2], theta24[nPar*i+4]], 
                              [theta24[nPar*i+4], theta24[nPar*i+3]]]))

    # subset Filter to contain only lag data for the validation set 
    Filter = Filter.iloc[len(train)-1:len(train)+len(validation)-1,:]
    
    # subset data to only include rows with available market cap data
    data = train.dropna()

    # Store parameters generated every week
    parBL = pd.DataFrame(np.zeros((len(validation), 10)), 
                                  columns=['Lambda', 'Pi_JCI', 'Pi_Bond', 'RegSwJCI', 
                                           'RegSwBond', 'omega', 'ExRetJCI', 'ExRetBond',
                                           'varJCI', 'varBond'])

    for t in range(0,len(validation)) :
        # compute Black-Litterman model parameters 
        # risk aversion coefficient Lambda_BL
        # lambda = (expected excess log-return + 0.5*variance of log return) / 
        # variance of log return of the equilibrium portfolio
        #Lambda_BL = (np.mean(np.log(1 + data['mktCapPortExRet'])) +\
        #             0.5*np.var(np.log(1+data['mktCapPortRet'])))/np.var(np.log(1+data['mktCapPortRet']))
        Lambda_BL = Lambda 
        #Lambda_BL = np.mean(np.log(1 + data.iloc[len(data)-103:len(data),:]['mktCapPortExRet']))/\
        #    np.var(np.log(1+data.iloc[len(data)-103:len(data),:]['mktCapPortRet']))
        parBL.loc[t,'Lambda'] = Lambda_BL

        # Sigma (variance covariance matrix of assets' log returns)
        sig = np.cov([np.log(1+data['JCIWeeklyRet']), np.log(1+data['BondWeeklyRet'])])
        #sig = np.cov([np.log(1+data.iloc[len(data)-103:len(data),:]['JCIWeeklyRet']), 
        #              np.log(1+data.iloc[len(data)-103:len(data),:]['BondWeeklyRet'])])
        
        # equilibrium weights w_eq from the markep cap weighted portfolio
        w_eq = np.array([data.iloc[len(data)-1,:]['w'], 1-data.iloc[len(data)-1,:]['w']])
    
        # implied equilibrium return Pi
        Pi = Lambda_BL*np.matmul(sig, w_eq)
        parBL.loc[t,'Pi_JCI'] = Pi[0]
        parBL.loc[t,'Pi_Bond'] = Pi[1]
    
        # create view portfolio (Q and P) from regime switching model
        # start week 1 by constructing assets' expected return from probability of regime 1 and 2 and prior of prev week
        xi = np.matrix(Filter[['p1', 'p2', 'p3', 'p4']].iloc[t])
        xi_next=p.T*xi.T
        ExRet = np.matrix(mu).T*xi_next
        parBL.loc[t,'RegSwJCI'] = ExRet[0,0]
        parBL.loc[t,'RegSwBond'] = ExRet[1,0]

        # construct view portfolio, with view on relative performance    
        if(ExRet[0,0] >= ExRet[1,0]) : 
            Q = ExRet[0,0] - ExRet[1,0]
            P = np.matrix([1, -1])
        elif(ExRet[0,0] < ExRet[1,0]) : 
            Q = ExRet[1,0] - ExRet[0,0]
            P = np.matrix([-1, 1])
        
        # compute uncertainty of view (omega = tau*P*Sigma*P')
        omega = tau*np.matmul(np.matmul(P, sig), P.T)
        parBL.loc[t,'omega'] = omega
    
        # compute assets' expected return and variance-covariance matrix using Black-Litterman equation
        vcov = np.linalg.inv(np.linalg.inv(tau*sig) + np.matmul(np.matmul(P.T, np.linalg.inv(omega)), P))
        parBL.loc[t,'varJCI'] = vcov[0,0]
        parBL.loc[t,'varBond'] = vcov[1,1]
    
        ExRet = np.matmul(vcov, (np.matmul(np.linalg.inv(tau*sig), Pi).reshape(2,1) +\
                                 np.matmul(P.T, np.linalg.inv(omega))*Q))
        parBL.loc[t,'ExRetJCI'] = ExRet[0,0]
        parBL.loc[t,'ExRetBond'] = ExRet[1,0]

        sigmaSq = np.diag(vcov)
    
        # set bounds for MVE optimisation (no shorting and total weights = 0.95)
        # set aside 5% of cash to pay transaction costs
        bounds = Bounds([0, 0], [0.95,0.95])
        linear_constraint = LinearConstraint([1, 1], [0], [0.95])
    
        # set initial weights
        w0=np.array([0.4, 0.4]) 
        # run constraint-bound optimisation to determine assets weight
        result = optimize.minimize(fun = utilLog, 
                                   x0 = w0, 
                                   args = (ExRet, sigmaSq, vcov, Lambda),
                                   method = 'trust-constr', 
                                   bounds = bounds,
                                   constraints = linear_constraint,
                                   options = {'disp': False})
        w_regSwitch4_BL = w_regSwitch4_BL.append(pd.Series(result.x, 
                                                           index=w_regSwitch4_BL.columns),
                                                 ignore_index=True)
    
        # update data for next period    
        data = data.append(validation.iloc[t,:])

    return w_regSwitch4_BL


# Function to compute portfolio weights based on Regime Switching model (4 regimes) 
# and Black-Litterman with absolute portfolio view  
def regSw4BLabs(optimVar) : 
    #---------------------------------------------------------------------------#
    # Asset allocation for regime switching model (4 regimes) + Black-Litterman #
    # with absolute view on assets' expected returns                            #
    #---------------------------------------------------------------------------#
    Lambda = optimVar[0]
    tau = optimVar[1]
    
    w_regSwitch4_BL = pd.DataFrame(columns=['JCI', 'Bond'])

    # use data in both train and validation set to estimate regime probabilities
    data = pd.DataFrame()
    data['JCILogRet'] = np.log(1+train['JCIExRet'])
    data['bondLogRet'] = np.log(1+train['BondExRet'])

    dataValidation = {'JCILogRet' : np.log(1+validation['JCIExRet']), 
                      'bondLogRet' : np.log(1+validation['BondExRet'])}
    data = data.append(pd.DataFrame(dataValidation))

    m = 4 # set number of regimes

    # import previously ran parameters
    theta24 = np.array(pd.read_csv("2var4regime_id_weekly_log.csv")['x'])

    # use the parameters to estimate filtered regime probabilities
    Filter = markovRegSwitch(theta24, data, m)[1]
    Filter['t index'] = list(range(0, len(data)))
    Filter['Regime 1'] = Filter['p1']>0.5
    Filter['Regime 2'] = Filter['p2']>0.5
    Filter['Regime 3'] = Filter['p3']>0.5
    Filter['Regime 4'] = Filter['p4']>0.5

    # construct probability transition matrix 
    nPar = int(len(theta24)/m)
    p = np.zeros((m,m)) 
    for i in range(0,m):
        for j in range(0,m-1):
            p[i][j]=theta24[nPar*i+5+j]
        p[i][(m-1)]=1-sum(p[i][0:(m-1)])

    # mean and variance-covariance matrix for each regime
    mu = [] 
    sig = []
    for i in range(0,m):
        mu.append(theta24[nPar*i:nPar*i+2])
        sig.append(np.matrix([[theta24[nPar*i+2], theta24[nPar*i+4]], 
                              [theta24[nPar*i+4], theta24[nPar*i+3]]]))

    # subset Filter to contain only lag data for the validation set 
    Filter = Filter.iloc[len(train)-1:len(train)+len(validation)-1,:]

    # subset data to only include rows with available market cap data
    data = train.dropna()

    # Store parameters generated every week
    parBL = pd.DataFrame(np.zeros((len(validation), 11)), 
                                  columns=['Lambda', 'Pi_JCI', 'Pi_Bond', 'RegSwJCI', 
                                           'RegSwBond', 'omegaJCI', 'omegaBond', 'ExRetJCI', 'ExRetBond',
                                           'varJCI', 'varBond'])

    for t in range(0,len(validation)) :
        # compute Black-Litterman model parameters 
        # risk aversion coefficient Lambda_BL
        # lambda = (expected excess log-return + 0.5*variance of log return) / 
        # variance of log return of the equilibrium portfolio
        #Lambda_BL = (np.mean(np.log(1 + data['mktCapPortExRet'])) +\
        #             0.5*np.var(np.log(1+data['mktCapPortRet'])))/np.var(np.log(1+data['mktCapPortRet']))
        Lambda_BL = Lambda 
        #Lambda_BL = np.mean(np.log(1 + data.iloc[len(data)-103:len(data),:]['mktCapPortExRet']))/\
        #    np.var(np.log(1+data.iloc[len(data)-103:len(data),:]['mktCapPortRet']))
        parBL.loc[t,'Lambda'] = Lambda_BL
        
        # Sigma (variance covariance matrix of assets' log returns)
        sig = np.cov([np.log(1+data['JCIWeeklyRet']), np.log(1+data['BondWeeklyRet'])])
        #sig = np.cov([np.log(1+data.iloc[len(data)-103:len(data),:]['JCIWeeklyRet']), 
        #              np.log(1+data.iloc[len(data)-103:len(data),:]['BondWeeklyRet'])])
    
        # equilibrium weights w_eq from the markep cap weighted portfolio
        w_eq = np.array([data.iloc[len(data)-1,:]['w'], 1-data.iloc[len(data)-1,:]['w']])
    
        # implied equilibrium return Pi
        Pi = Lambda_BL*np.matmul(sig, w_eq)
        parBL.loc[t,'Pi_JCI'] = Pi[0]
        parBL.loc[t,'Pi_Bond'] = Pi[1]
    
        # create view portfolio (Q and P) from regime switching model
        # start week 1 by constructing assets' expected return from probability of regime 1 and 2 and prior of prev week
        xi = np.matrix(Filter[['p1', 'p2', 'p3', 'p4']].iloc[t])
        xi_next=p.T*xi.T
        ExRet = np.matrix(mu).T*xi_next
        parBL.loc[t,'RegSwJCI'] = ExRet[0,0]
        parBL.loc[t,'RegSwBond'] = ExRet[1,0]
    
        # construct absolute view portfolio based on the regime switching model expected return
        Q = ExRet
        P = np.matrix([[1, 0], [0,1]])
    
        # compute uncertainty of views using variance from regime switching
        vcov_regSwitch = sig[0]*float(xi_next[0]) + sig[1]*float(xi_next[1])
        omega = np.diag(vcov_regSwitch)
        #omega = tau*np.matmul(np.matmul(P, sig), P.T)
        parBL.loc[t,'omegaJCI'] = omega[0,0]
        parBL.loc[t,'omegaBond'] = omega[1,1]
    
        # compute assets' expected return and variance-covariance matrix using Black-Litterman equation
        vcov = np.linalg.inv(np.linalg.inv(tau*sig) + np.matmul(np.matmul(P.T, np.linalg.inv(omega)), P))
        parBL.loc[t,'varJCI'] = vcov[0,0]
        parBL.loc[t,'varBond'] = vcov[1,1]
    
        ExRet = np.matmul(vcov, (np.matmul(np.linalg.inv(tau*sig), Pi).reshape(2,1) +\
                                 np.matmul(P.T, np.linalg.inv(omega))*Q))
        parBL.loc[t,'ExRetJCI'] = ExRet[0,0]
        parBL.loc[t,'ExRetBond'] = ExRet[1,0]

        sigmaSq = np.diag(vcov)
    
        # set bounds for MVE optimisation (no shorting and total weights = 0.95)
        # set aside 5% of cash to pay transaction costs
        bounds = Bounds([0, 0], [0.95,0.95])
        linear_constraint = LinearConstraint([1, 1], [0], [0.95])
    
        # set initial weights
        w0=np.array([0.4, 0.4]) 
        # run constraint-bound optimisation to determine assets weight
        result = optimize.minimize(fun = utilLog, 
                                   x0 = w0, 
                                   args = (ExRet, sigmaSq, vcov, Lambda),
                                   method = 'trust-constr', 
                                   bounds = bounds,
                                   constraints = linear_constraint,
                                   options = {'disp': False})
        w_regSwitch4_BL = w_regSwitch4_BL.append(pd.Series(result.x, 
                                                           index=w_regSwitch4_BL.columns),
                                                 ignore_index=True)
    
        # update data for next period    
        data = data.append(validation.iloc[t,:])
        
    return w_regSwitch4_BL


# Function to compute portfolio weights based on Regime Switching model (2 regimes) 
# with Transaction Cost in the MVO algorithm
def regSw4_TC(Lambda) : 
    #---------------------------------------------------------#
    # Asset allocation for regime switching model (4 regimes) #
    # including transaction costs                             #
    #---------------------------------------------------------#

    w_regSwitch4_TC = pd.DataFrame(columns=['JCI', 'Bond'])

    # use data in both train and validation set to estimate regime probabilities
    data = pd.DataFrame()
    data['JCILogRet'] = np.log(1+train['JCIExRet'])
    data['bondLogRet'] = np.log(1+train['BondExRet'])

    dataValidation = {'JCILogRet' : np.log(1+validation['JCIExRet']), 
                      'bondLogRet' : np.log(1+validation['BondExRet'])}
    data = data.append(pd.DataFrame(dataValidation))

    m = 4 # set number of regimes
    
    # import previously ran parameters
    theta24 = np.array(pd.read_csv("2var4regime_id_weekly_log.csv")['x'])

    # use the parameters to estimate filtered regime probabilities
    Filter = markovRegSwitch(theta24, data, m)[1]
    Filter['t index'] = list(range(0, len(data)))
    Filter['Regime 1'] = Filter['p1']>0.5
    Filter['Regime 2'] = Filter['p2']>0.5
    Filter['Regime 3'] = Filter['p3']>0.5
    Filter['Regime 4'] = Filter['p4']>0.5
    
    # construct probability transition matrix 
    nPar = int(len(theta24)/m)
    p = np.zeros((m,m)) 
    for i in range(0,m):
        for j in range(0,m-1):
            p[i][j]=theta24[nPar*i+5+j]
        p[i][(m-1)]=1-sum(p[i][0:(m-1)])

    # mean and variance-covariance matrix for each regime
    mu = [] 
    sig = []
    for i in range(0,m):
        mu.append(theta24[nPar*i:nPar*i+2]) 
        sig.append(np.matrix([[theta24[nPar*i+2], theta24[nPar*i+4]], 
                              [theta24[nPar*i+4], theta24[nPar*i+3]]]))

    # subset Filter to contain only lag data for the validation set 
    Filter = Filter.iloc[len(train)-1:len(train)+len(validation)-1,:]
    
    # set weights at t=0 to be 0
    w_prev = np.array([0, 0]) 
    # array of bid and ask costs, assuming same unit costs for all assets
    askArray = np.array([ask, ask])
    bidArray = np.array([bid, bid])

    for t in range(0,len(validation)) :
        # start week 1 by constructing expected mean and var-cov matrix 
        # from probabilites of regime 1 and 2 and prior of prev week
        xi = np.matrix(Filter[['p1', 'p2', 'p3', 'p4']].iloc[t])
        xi_next=p.T*xi.T
        
        ExRet = np.matrix(mu).T*xi_next
        vcov = sig[0]*float(xi_next[0]) + sig[1]*float(xi_next[1]) + \
            sig[2]*float(xi_next[2]) + sig[3]*float(xi_next[3])
        sigmaSq = np.diag(vcov)
    
        # set bounds for MVE optimisation (no shorting and total weights = 0.95)
        # set aside 5% of cash to pay transaction costs
        bounds = Bounds([0, 0], [0.95,0.95])
        linear_constraint = LinearConstraint([1, 1], [0], [0.95])
    
        # set initial weights
        w0=np.array([0.4, 0.4]) 
        # run constraint-bound optimisation to determine assets weight
        result = optimize.minimize(fun = utilLogTC, 
                                   x0 = w0, 
                                   args = (w_prev, askArray, bidArray, ExRet, sigmaSq, vcov, Lambda),
                                   method = 'trust-constr', 
                                   bounds = bounds,
                                   constraints = linear_constraint,
                                   options = {'disp': False})
        w_regSwitch4_TC = w_regSwitch4_TC.append(pd.Series(result.x, 
                                                           index=w_regSwitch4_TC.columns), 
                                                 ignore_index=True)
        
        # update data for next period    
        w_prev = result.x

    return w_regSwitch4_TC


# Function to compute portfolio weights based on Regime Switching model (4 regimes) 
# and Black-Litterman with absolute portfolio view  
# with Transaction Cost in the MVO algorithm
def regSw4BLabs_TC(optimVar) : 
    #---------------------------------------------------------------------------#
    # Asset allocation for regime switching model (4 regimes) + Black-Litterman #
    # with absolute view on assets' expected returns and                        #
    # including transaction costs                                               #
    #---------------------------------------------------------------------------#
    Lambda = optimVar[0]
    tau = optimVar[1]
    
    w_regSwitch4_BL_TC = pd.DataFrame(columns=['JCI', 'Bond'])

    # use data in both train and validation set to estimate regime probabilities
    data = pd.DataFrame()
    data['JCILogRet'] = np.log(1+train['JCIExRet'])
    data['bondLogRet'] = np.log(1+train['BondExRet'])

    dataValidation = {'JCILogRet' : np.log(1+validation['JCIExRet']), 
                      'bondLogRet' : np.log(1+validation['BondExRet'])}
    data = data.append(pd.DataFrame(dataValidation))

    m = 4 # set number of regimes

    # import previously ran parameters
    theta24 = np.array(pd.read_csv("2var4regime_id_weekly_log.csv")['x'])

    # use the parameters to estimate filtered regime probabilities
    Filter = markovRegSwitch(theta24, data, m)[1]
    Filter['t index'] = list(range(0, len(data)))
    Filter['Regime 1'] = Filter['p1']>0.5
    Filter['Regime 2'] = Filter['p2']>0.5
    Filter['Regime 3'] = Filter['p3']>0.5
    Filter['Regime 4'] = Filter['p4']>0.5

    # construct probability transition matrix 
    nPar = int(len(theta24)/m)
    p = np.zeros((m,m)) 
    for i in range(0,m):
        for j in range(0,m-1):
            p[i][j]=theta24[nPar*i+5+j]
        p[i][(m-1)]=1-sum(p[i][0:(m-1)])

    # mean and variance-covariance matrix for each regime
    mu = [] 
    sig = []
    for i in range(0,m):
        mu.append(theta24[nPar*i:nPar*i+2])
        sig.append(np.matrix([[theta24[nPar*i+2], theta24[nPar*i+4]], 
                              [theta24[nPar*i+4], theta24[nPar*i+3]]]))

    # subset Filter to contain only lag data for the validation set 
    Filter = Filter.iloc[len(train)-1:len(train)+len(validation)-1,:]

    # subset data to only include rows with available market cap data
    data = train.dropna()

    # Store parameters generated every week
    parBL = pd.DataFrame(np.zeros((len(validation), 11)), 
                                  columns=['Lambda', 'Pi_JCI', 'Pi_Bond', 'RegSwJCI', 
                                           'RegSwBond', 'omegaJCI', 'omegaBond', 'ExRetJCI', 'ExRetBond',
                                           'varJCI', 'varBond'])
    
    # set weights at t=0 to be 0
    w_prev = np.array([0, 0]) 
    # array of bid and ask costs, assuming same unit costs for all assets
    askArray = np.array([ask, ask])
    bidArray = np.array([bid, bid])

    for t in range(0,len(validation)) :
        # compute Black-Litterman model parameters 
        # risk aversion coefficient Lambda_BL
        # lambda = (expected excess log-return + 0.5*variance of log return) / 
        # variance of log return of the equilibrium portfolio
        #Lambda_BL = (np.mean(np.log(1 + data['mktCapPortExRet'])) +\
        #             0.5*np.var(np.log(1+data['mktCapPortRet'])))/np.var(np.log(1+data['mktCapPortRet']))
        Lambda_BL = Lambda 
        #Lambda_BL = np.mean(np.log(1 + data.iloc[len(data)-103:len(data),:]['mktCapPortExRet']))/\
        #    np.var(np.log(1+data.iloc[len(data)-103:len(data),:]['mktCapPortRet']))
        parBL.loc[t,'Lambda'] = Lambda_BL
        
        # Sigma (variance covariance matrix of assets' log returns)
        sig = np.cov([np.log(1+data['JCIWeeklyRet']), np.log(1+data['BondWeeklyRet'])])
        #sig = np.cov([np.log(1+data.iloc[len(data)-103:len(data),:]['JCIWeeklyRet']), 
        #              np.log(1+data.iloc[len(data)-103:len(data),:]['BondWeeklyRet'])])
    
        # equilibrium weights w_eq from the markep cap weighted portfolio
        w_eq = np.array([data.iloc[len(data)-1,:]['w'], 1-data.iloc[len(data)-1,:]['w']])
    
        # implied equilibrium return Pi
        Pi = Lambda_BL*np.matmul(sig, w_eq)
        parBL.loc[t,'Pi_JCI'] = Pi[0]
        parBL.loc[t,'Pi_Bond'] = Pi[1]
    
        # create view portfolio (Q and P) from regime switching model
        # start week 1 by constructing assets' expected return from probability of regime 1 and 2 and prior of prev week
        xi = np.matrix(Filter[['p1', 'p2', 'p3', 'p4']].iloc[t])
        xi_next=p.T*xi.T
        ExRet = np.matrix(mu).T*xi_next
        parBL.loc[t,'RegSwJCI'] = ExRet[0,0]
        parBL.loc[t,'RegSwBond'] = ExRet[1,0]
    
        # construct absolute view portfolio based on the regime switching model expected return
        Q = ExRet
        P = np.matrix([[1, 0], [0,1]])
    
        # compute uncertainty of views using variance from regime switching
        vcov_regSwitch = sig[0]*float(xi_next[0]) + sig[1]*float(xi_next[1])
        omega = np.diag(vcov_regSwitch)
        #omega = tau*np.matmul(np.matmul(P, sig), P.T)
        parBL.loc[t,'omegaJCI'] = omega[0,0]
        parBL.loc[t,'omegaBond'] = omega[1,1]
    
        # compute assets' expected return and variance-covariance matrix using Black-Litterman equation
        vcov = np.linalg.inv(np.linalg.inv(tau*sig) + np.matmul(np.matmul(P.T, np.linalg.inv(omega)), P))
        parBL.loc[t,'varJCI'] = vcov[0,0]
        parBL.loc[t,'varBond'] = vcov[1,1]
    
        ExRet = np.matmul(vcov, (np.matmul(np.linalg.inv(tau*sig), Pi).reshape(2,1) +\
                                 np.matmul(P.T, np.linalg.inv(omega))*Q))
        parBL.loc[t,'ExRetJCI'] = ExRet[0,0]
        parBL.loc[t,'ExRetBond'] = ExRet[1,0]

        sigmaSq = np.diag(vcov)
    
        # set bounds for MVE optimisation (no shorting and total weights = 0.95)
        # set aside 5% of cash to pay transaction costs
        bounds = Bounds([0, 0], [0.95,0.95])
        linear_constraint = LinearConstraint([1, 1], [0], [0.95])
    
        # set initial weights
        w0=np.array([0.4, 0.4]) 
        # run constraint-bound optimisation to determine assets weight
        result = optimize.minimize(fun = utilLogTC, 
                                   x0 = w0, 
                                   args = (w_prev, askArray, bidArray, ExRet, sigmaSq, vcov, Lambda),
                                   method = 'trust-constr', 
                                   bounds = bounds,
                                   constraints = linear_constraint,
                                   options = {'disp': False})
        w_regSwitch4_BL_TC = w_regSwitch4_BL_TC.append(pd.Series(result.x, 
                                                           index=w_regSwitch4_BL_TC.columns),
                                                 ignore_index=True)
    
        # update data for next period    
        data = data.append(validation.iloc[t,:])
        w_prev = result.x
        
    return w_regSwitch4_BL_TC
  

# function to plot heatmap
def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=False, bottom=True,
                   labeltop=False, labelbottom=True)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


# function to annotate data in the heatmap
def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


# Function to compute portfolio weights based on Mean-Variance Efficient Portfolio 
# for out-sample data
def MVE_outSample(Lambda) : 
    #---------------------------------------------------------------#
    # Asset allocation for benchmark MVE model for out-sample data  #
    #---------------------------------------------------------------#
    # compute log returns for both train and validation data set
    data = pd.DataFrame()
    data['JCILogRet'] = np.log(1+train['JCIExRet'])
    data['bondLogRet'] = np.log(1+train['BondExRet'])
    
    dataValidation = {'JCILogRet' : np.log(1+validation['JCIExRet']), 
                      'bondLogRet' : np.log(1+validation['BondExRet'])}
    data = data.append(pd.DataFrame(dataValidation))
    
    w = pd.DataFrame(columns=['JCI', 'Bond'])

    for t in range(0,len(test)) :
        # compute assets expected returns, variance-covariance matrix and 
        # vector of variances using all historical data up to last week
        ExRet = np.array([np.mean(data['JCILogRet']),
                          np.mean(data['bondLogRet'])])
        vcov = np.cov(data['JCILogRet'], data['bondLogRet'])
        sigmaSq = np.diag(vcov)
        
        # set bounds for MVE optimisation (no shorting and total weights = 0.95)
        # set aside 5% of cash to pay transaction costs
        bounds = Bounds([0, 0], [0.95,0.95])
        linear_constraint = LinearConstraint([1, 1], [0], [0.95])
        
        # set initial guess of the weights
        w0=np.array([0.4, 0.4]) 
        # run constraint-bound optimisation to determine assets weight
        result = optimize.minimize(fun = utilLog, 
                                   x0 = w0, 
                                   args = (ExRet, sigmaSq, vcov, Lambda),
                                   method = 'trust-constr', 
                                   bounds = bounds,
                                   constraints = linear_constraint,
                                   options = {'disp': False})
        w = w.append(pd.Series(result.x, index=w.columns), ignore_index=True)
        
        # update data for next period    
        data = data.append(pd.Series([np.log(1+test['JCIExRet'].iloc[t]),
                                      np.log(1+test['BondExRet'].iloc[t])], index=data.columns), 
                           ignore_index=True)
    return w


# Function to compute portfolio weights based on Regime Switching model (2 regimes) 
# and Black-Litterman with absolute portfolio view
def regSw2BLabs_outSample(optimVar) :
    #---------------------------------------------------------------------------#
    # Asset allocation for regime switching model (2 regimes) + Black-Litterman #
    # with absolute view on assets' expected returns for out-sample test        #
    #---------------------------------------------------------------------------#
    # assets' expected returns from markov regime switching model are used to create 
    # view portfolio in Black-Litterman model. Equilibrium weights (prior weights) 
    # are computed from the market cap of both assets at the end of previous week 
    Lambda = optimVar[0]
    tau = optimVar[1]
    
    w_regSwitch2_BL = pd.DataFrame(columns=['JCI', 'Bond'])

    # use all data to estimate regime probabilities
    data = pd.DataFrame()
    data['JCILogRet'] = np.log(1+data_weekly['JCIExRet'])
    data['bondLogRet'] = np.log(1+data_weekly['BondExRet'])

    m = 2 # set number of regimes

    # import previously ran parameters
    theta22 = np.array(pd.read_csv("2var2regime_id_weekly_log.csv")['x'])

    # use the parameters to estimate filtered regime probabilities
    Filter = markovRegSwitch(theta22, data, m)[1]
    Filter['t index'] = list(range(0, len(data)))
    Filter['Regime 1'] = Filter['p1']>0.5
    Filter['Regime 2'] = Filter['p2']>0.5

    # construct probability transition matrix 
    p = np.matrix([[theta22[5], 1-theta22[5]], 
                   [theta22[11], 1-theta22[11]]])

    # mean and variance-covariance matrix for each regime
    nPar = int(len(theta22)/m)
    mu = [] 
    sig = []
    for i in range(0,m):
        mu.append(theta22[nPar*i:nPar*i+2])
        sig.append(np.matrix([[theta22[nPar*i+2], theta22[nPar*i+4]], 
                              [theta22[nPar*i+4], theta22[nPar*i+3]]]))

    # subset Filter to contain only lag data for the validation set 
    Filter = Filter.iloc[len(train)+len(validation)-1:len(train)+len(validation)+len(test)-1,:]

    # subset data to only include rows with available market cap data
    data = train.dropna()
    data = data.append(validation)

    # Store parameters generated every week
    parBL = pd.DataFrame(np.zeros((len(validation), 11)), 
                                  columns=['Lambda', 'Pi_JCI', 'Pi_Bond', 'RegSwJCI', 
                                           'RegSwBond', 'omegaJCI', 'omegaBond', 'ExRetJCI', 'ExRetBond',
                                           'varJCI', 'varBond'])

    for t in range(0,len(test)) :
        # compute Black-Litterman model parameters 
        # risk aversion coefficient Lambda_BL
        # lambda = (expected excess log-return + 0.5*variance of log return) / 
        # variance of log return of the equilibrium portfolio
        #Lambda_BL = (np.mean(np.log(1 + data['mktCapPortExRet'])) +\
        #             0.5*np.var(np.log(1+data['mktCapPortRet'])))/np.var(np.log(1+data['mktCapPortRet']))
        Lambda_BL = Lambda 
        #Lambda_BL = np.mean(np.log(1 + data.iloc[len(data)-103:len(data),:]['mktCapPortExRet']))/\
        #    np.var(np.log(1+data.iloc[len(data)-103:len(data),:]['mktCapPortRet']))
        parBL.loc[t,'Lambda'] = Lambda_BL
        
        # Sigma (variance covariance matrix of assets' log returns)
        sig = np.cov([np.log(1+data['JCIWeeklyRet']), np.log(1+data['BondWeeklyRet'])])
        #sig = np.cov([np.log(1+data.iloc[len(data)-103:len(data),:]['JCIWeeklyRet']), 
        #              np.log(1+data.iloc[len(data)-103:len(data),:]['BondWeeklyRet'])])
        
        # equilibrium weights w_eq from the markep cap weighted portfolio
        w_eq = np.array([data.iloc[len(data)-1,:]['w'], 1-data.iloc[len(data)-1,:]['w']])
    
        # implied equilibrium return Pi
        Pi = Lambda_BL*np.matmul(sig, w_eq)
        parBL.loc[t,'Pi_JCI'] = Pi[0]
        parBL.loc[t,'Pi_Bond'] = Pi[1]
    
        # create view portfolio (Q and P) from regime switching model
        # start week 1 by constructing assets' expected return from probability of regime 1 and 2 and prior of prev week
        xi = np.matrix(Filter[['p1', 'p2']].iloc[t])
        xi_next=p.T*xi.T
        ExRet = np.matrix(mu).T*xi_next
        parBL.loc[t,'RegSwJCI'] = ExRet[0,0]
        parBL.loc[t,'RegSwBond'] = ExRet[1,0]
            
        # construct absolute view portfolio based on the regime switching model expected return
        Q = ExRet
        P = np.matrix([[1, 0], [0,1]])
        
        # compute uncertainty of views using variance from regime switching
        vcov_regSwitch = sig[0]*float(xi_next[0]) + sig[1]*float(xi_next[1])
        omega = np.diag(vcov_regSwitch)
        #omega = tau*np.matmul(np.matmul(P, sig), P.T)
        parBL.loc[t,'omegaJCI'] = omega[0,0]
        parBL.loc[t,'omegaBond'] = omega[1,1]
        
        # compute assets' expected return and variance-covariance matrix using Black-Litterman equation
        vcov = np.linalg.inv(np.linalg.inv(tau*sig) + np.matmul(np.matmul(P.T, np.linalg.inv(omega)), P))
        parBL.loc[t,'varJCI'] = vcov[0,0]
        parBL.loc[t,'varBond'] = vcov[1,1]
    
        ExRet = np.matmul(vcov, (np.matmul(np.linalg.inv(tau*sig), Pi).reshape(2,1) +\
                                 np.matmul(P.T, np.linalg.inv(omega))*Q))
        parBL.loc[t,'ExRetJCI'] = ExRet[0,0]
        parBL.loc[t,'ExRetBond'] = ExRet[1,0]

        sigmaSq = np.diag(vcov)
    
        # set bounds for MVE optimisation (no shorting and total weights = 0.95)
        # set aside 5% of cash to pay transaction costs
        bounds = Bounds([0, 0], [0.95,0.95])
        linear_constraint = LinearConstraint([1, 1], [0], [0.95])
    
        # set initial weights
        w0=np.array([0.4, 0.4]) 
        # run constraint-bound optimisation to determine assets weight
        # Lambda = Lambda_BL # set Lambda in MVE equals to Lambda implied by equilibrium portfolio
        result = optimize.minimize(fun = utilLog, 
                                   x0 = w0, 
                                   args = (ExRet, sigmaSq, vcov, Lambda),
                                   method = 'trust-constr', 
                                   bounds = bounds,
                                   constraints = linear_constraint,
                                   options = {'disp': False})
        w_regSwitch2_BL = w_regSwitch2_BL.append(pd.Series(result.x, 
                                                           index=w_regSwitch2_BL.columns),
                                                 ignore_index=True)
    
        # update data for next period    
        data = data.append(test.iloc[t,:])

    return w_regSwitch2_BL


# Function to compute portfolio weights based on Regime Switching model (2 regimes) 
# with Transaction Cost in the MVO algorithm
def regSw4_TC_outSample(Lambda) : 
    #---------------------------------------------------------#
    # Asset allocation for regime switching model (4 regimes) #
    # including transaction costs for out-sample test         #
    #---------------------------------------------------------#

    w_regSwitch4_TC = pd.DataFrame(columns=['JCI', 'Bond'])

    # use all data to estimate regime probabilities
    data = pd.DataFrame()
    data['JCILogRet'] = np.log(1+data_weekly['JCIExRet'])
    data['bondLogRet'] = np.log(1+data_weekly['BondExRet'])

    m = 4 # set number of regimes
    
    # import previously ran parameters
    theta24 = np.array(pd.read_csv("2var4regime_id_weekly_log.csv")['x'])

    # use the parameters to estimate filtered regime probabilities
    Filter = markovRegSwitch(theta24, data, m)[1]
    Filter['t index'] = list(range(0, len(data)))
    Filter['Regime 1'] = Filter['p1']>0.5
    Filter['Regime 2'] = Filter['p2']>0.5
    Filter['Regime 3'] = Filter['p3']>0.5
    Filter['Regime 4'] = Filter['p4']>0.5
    
    # construct probability transition matrix 
    nPar = int(len(theta24)/m)
    p = np.zeros((m,m)) 
    for i in range(0,m):
        for j in range(0,m-1):
            p[i][j]=theta24[nPar*i+5+j]
        p[i][(m-1)]=1-sum(p[i][0:(m-1)])

    # mean and variance-covariance matrix for each regime
    mu = [] 
    sig = []
    for i in range(0,m):
        mu.append(theta24[nPar*i:nPar*i+2]) 
        sig.append(np.matrix([[theta24[nPar*i+2], theta24[nPar*i+4]], 
                              [theta24[nPar*i+4], theta24[nPar*i+3]]]))

    # subset Filter to contain only lag data for the validation set 
    Filter = Filter.iloc[len(train)+len(validation)-1:len(train)+len(validation)+len(test)-1,:]
    
    # set weights at t=0 to be 0
    w_prev = np.array([0, 0]) 
    # array of bid and ask costs, assuming same unit costs for all assets
    askArray = np.array([ask, ask])
    bidArray = np.array([bid, bid])

    for t in range(0,len(test)) :
        # start week 1 by constructing expected mean and var-cov matrix 
        # from probabilites of regime 1 and 2 and prior of prev week
        xi = np.matrix(Filter[['p1', 'p2', 'p3', 'p4']].iloc[t])
        xi_next=p.T*xi.T
        
        ExRet = np.matrix(mu).T*xi_next
        vcov = sig[0]*float(xi_next[0]) + sig[1]*float(xi_next[1]) + \
            sig[2]*float(xi_next[2]) + sig[3]*float(xi_next[3])
        sigmaSq = np.diag(vcov)
    
        # set bounds for MVE optimisation (no shorting and total weights = 0.95)
        # set aside 5% of cash to pay transaction costs
        bounds = Bounds([0, 0], [0.95,0.95])
        linear_constraint = LinearConstraint([1, 1], [0], [0.95])
    
        # set initial weights
        w0=np.array([0.4, 0.4]) 
        # run constraint-bound optimisation to determine assets weight
        result = optimize.minimize(fun = utilLogTC, 
                                   x0 = w0, 
                                   args = (w_prev, askArray, bidArray, ExRet, sigmaSq, vcov, Lambda),
                                   method = 'trust-constr', 
                                   bounds = bounds,
                                   constraints = linear_constraint,
                                   options = {'disp': False})
        w_regSwitch4_TC = w_regSwitch4_TC.append(pd.Series(result.x, 
                                                           index=w_regSwitch4_TC.columns), 
                                                 ignore_index=True)
        
        # update data for next period    
        w_prev = result.x

    return w_regSwitch4_TC

# import and format data
data_weekly = pd.read_csv("weeklyret.csv")
data_weekly['date'] = pd.to_datetime(data_weekly['date'],format='%m/%d/%Y')

# import market cap data for Black-Litterman prior
mktCap = pd.read_excel(r'Market Cap.xlsx', skiprows=1)

JCIMktCap = pd.DataFrame({'date' : mktCap.iloc[:,4], 'JCI Market Cap' : mktCap.iloc[:,5]})
# fix error on JCI data
JCIMktCap.loc[JCIMktCap['JCI Market Cap']>10**10, 'JCI Market Cap'] = \
    JCIMktCap.loc[JCIMktCap['JCI Market Cap']>10**10, 'JCI Market Cap'] - 5*10**10

# since market cap data on ABTRINDO is not available, we use market cap data of BINDO
# correlation between ABTRINDO and BINDO daily price is 0.99988, so this is acceptable
BondMktCap = pd.DataFrame({'date' : mktCap.iloc[:,7], 'Bond Market Cap' : mktCap.iloc[:,8]/10**6})
BondMktCap = BondMktCap.dropna() # drop rows with missing data at the end of the df

# merge market cap data with weekly return data
data_weekly = data_weekly.merge(JCIMktCap, how='left', on='date')
data_weekly = data_weekly.merge(BondMktCap, how='left', on='date')

# create market cap weighted portfolio 
# create weights based on market cap of JCI and bond every week 
data_weekly['w'] = data_weekly['JCI Market Cap']/(data_weekly['JCI Market Cap'] + data_weekly['Bond Market Cap'])
# compute market cap weighted portfolio return
data_weekly['mktCapPortRet'] = data_weekly['w']*data_weekly['JCIWeeklyRet'] +\
    (1-data_weekly['w'])*data_weekly['BondWeeklyRet']
# compute cumulative returna and excess return
data_weekly['mktCapPortExRet'] = data_weekly['mktCapPortRet'] - data_weekly['Rf_weekly']

# split into train, validation and test data set 
train = data_weekly.loc[data_weekly['year']<=2010,]
validation = data_weekly.loc[(data_weekly['year']>2010) & (data_weekly['year']<=2015),]
test = data_weekly.loc[data_weekly['year']>2015,]

# set parameters for transacation cost
bid = 1.05/100 # cost for buying in % of transaction value
ask = 1.15/100 # cost for selling in % of transaction value


#----------------------------------------------------------------------------#
# Manual search of optimum tau values for regime-switching + Black-Litterman #
#----------------------------------------------------------------------------#
# assume risk aversion coefficient
Lambda = 4

# set list of tau values to be tested
tau_list = np.arange(0.01, 0.1, 0.01)
tau_list = np.concatenate((tau_list, np.arange(0.1, 0.5, 0.02)))
tau_list = np.concatenate((tau_list, np.arange(0.5, 1, 0.1)))
#tau_list = np.concatenate((tau_list, np.arange(1, 11, 1)))

portCumRet2_TC = np.zeros((1, len(validation)))
portCumRet4_TC = np.zeros((1, len(validation)))
SR2_TC = []
SR4_TC = []

# compute portfolio cumulative returns and Sharpe Ratio for each tau value
# use function '....._TC()' for MVO algorithm with TC 
# functions without '_TC' indicates standard MVO algorithm
for n in range(0, len(tau_list)) : 
    print(tau_list[n])
    
    w_regSwitch2_BL_TC = regSw2BLabs_TC(np.array([Lambda, tau_list[n]]))
    port2TC = portCumRetTC(w_regSwitch2_BL_TC)
    portCumRet2_TC = np.vstack((portCumRet2_TC, port2TC[0]))
    SR2_TC.append(port2TC[1])
    
    w_regSwitch4_BL_TC = regSw4BLabs_TC(np.array([Lambda, tau_list[n]]))
    port4TC = portCumRetTC(w_regSwitch4_BL_TC)
    portCumRet4_TC = np.vstack((portCumRet4_TC, port4TC[0]))
    SR4_TC.append(port4TC[1])
    

# compare sharpe ratio
plt.figure(figsize=(10, 6))
plt.plot(tau_list, SR2_TC, label="Regime Switching (2 regimes) + Black-Litterman")
plt.plot(tau_list, SR4_TC, label="Regime Switching (4 regimes) + Black-Litterman")
plt.title('Sharpe Ratio vs Tau')
plt.ylabel('Sharpe Ratio')
plt.xlabel('tau')
plt.legend(loc='upper right')

# compare portfolio cumulative returns at t=T
plt.figure(figsize=(10, 6))
plt.plot(tau_list, portCumRet2_TC[1:,254], label="Regime Switching (2 regimes) + Black-Litterman")
plt.plot(tau_list, portCumRet4_TC[1:,254], label="Regime Switching (4 regimes) + Black-Litterman")
plt.title('Portfolio Cumulative Return at t=T vs Tau')
plt.ylabel('Cumulative Returns')
plt.xlabel('tau')
plt.legend(loc='upper right')

# get values of tau that maximises portfolio cumulative returns and Sharpe Ratio
tau_max_2 = tau_list[portCumRet2_TC[1:,254] == max(portCumRet2_TC[1:,254])]
tau_max_2 = tau_list[SR2_TC == max(SR2_TC)]

tau_max_4 = tau_list[portCumRet4_TC[1:,254] == max(portCumRet4_TC[1:,254])]
tau_max_4 = tau_list[SR4_TC == max(SR4_TC)]

# compute portfolio cumulative returns for MVE and regime switching models with TC
w_TC = MVE_TC(Lambda)
portCumRet_TC = portCumRetTC(w_TC)[0]
w_regSwitch2_TC = regSw2_TC(Lambda)
portCumRetRegSwitch2_TC = portCumRetTC(w_regSwitch2_TC)[0]
w_regSwitch4_TC = regSw4_TC(Lambda)
portCumRetRegSwitch4_TC = portCumRetTC(w_regSwitch4_TC)[0]

# compute portfolio cumulative returns for regime switching + Black-Litterman models with TC and tau_max
w_regSwitch2_BL_TC = regSw2BLabs_TC(np.array([Lambda, tau_max_2]))
portCumRetRegSwitch2_BL_TC = portCumRetTC(w_regSwitch2_BL_TC)[0]
w_regSwitch4_BL_TC = regSw4BLabs_TC(np.array([Lambda, tau_max_4]))
portCumRetRegSwitch4_BL_TC = portCumRetTC(w_regSwitch4_BL_TC)[0]

# compare cumulative returns for all models
plt.figure(figsize=(10, 6))
plt.plot(validation['date'], portCumRet_TC, label='Mean-Variance Efficient Portfolio after TC')
plt.plot(validation['date'], portCumRetRegSwitch2_TC, label='Regime Switching (2 regimes) after TC')
plt.plot(validation['date'], portCumRetRegSwitch4_TC, label='Regime Switching (4 regimes) after TC')
plt.plot(validation['date'], portCumRetRegSwitch2_BL_TC, label='Regime Switching (2 regimes) + Black-Litterman (tau=0.7) after TC')
plt.plot(validation['date'], portCumRetRegSwitch4_BL_TC, label='Regime Switching (4 regimes) + Black-Litterman (tau=0.8) after TC')
plt.ylabel('Cumulative Return')
plt.legend(loc='upper left')


#----------------------------------------------------------------------------#
# Manual search of optimum lambda values for MVE and regime-switching models #
#----------------------------------------------------------------------------#
# set list of lambda values to be tested
lambda_list = np.arange(0.2, 1, 0.2)
lambda_list = np.concatenate((lambda_list, np.arange(1, 5, 0.5)))
lambda_list = np.concatenate((lambda_list, np.arange(5, 21, 1)))

portCumRet_MVE = np.zeros((1, len(validation)))
portCumRet_regSw2 = np.zeros((1, len(validation)))
portCumRet_regSw4 = np.zeros((1, len(validation)))
SR_MVE = []
SR_regSw2 = []
SR_regSw4 = []

# compute portfolio cumulative returns and Sharpe Ratio for each Lambda value
# use function '....._TC()' for MVO algorithm with TC 
# functions without '_TC' indicates standard MVO algorithm
for n in range(0, len(lambda_list)) : 
    print(lambda_list[n])
    
    w = MVE(lambda_list[n])
    #w = MVE_TC(lambda_list[n]) 
    port = portCumRetTC(w)
    portCumRet_MVE = np.vstack((portCumRet_MVE, port[0]))
    SR_MVE.append(port[1])
    
    w_regSwitch2 = regSw2(lambda_list[n])
    port2TC = portCumRetTC(w_regSwitch2)
    #w_regSwitch2_TC = regSw2_TC(lambda_list[n])
    #port2TC = portCumRetTC(w_regSwitch2_TC)
    portCumRet_regSw2 = np.vstack((portCumRet_regSw2, port2TC[0]))
    SR_regSw2.append(port2TC[1])
    
    w_regSwitch4 = regSw4(lambda_list[n])
    port4TC = portCumRetTC(w_regSwitch4)
    #w_regSwitch4_TC = regSw4_TC(lambda_list[n])
    #port4TC = portCumRetTC(w_regSwitch4_TC)
    portCumRet_regSw4 = np.vstack((portCumRet_regSw4, port4TC[0]))
    SR_regSw4.append(port4TC[1])

portCumRet_MVE = portCumRet_MVE[1:,:]
portCumRet_regSw2 = portCumRet_regSw2[1:,:]
portCumRet_regSw4 = portCumRet_regSw4[1:,:]

# compare portfolio cumulative returns at t=T
plt.figure(figsize=(10, 6))
plt.plot(lambda_list, portCumRet_MVE[:,254], label="Mean-Variance Efficient Portfolio")
plt.plot(lambda_list, portCumRet_regSw2[:,254], label="Regime Switching (2 regimes)")
plt.plot(lambda_list, portCumRet_regSw4[:,254], label="Regime Switching (4 regimes)")
plt.title('Portfolio Cumulative Return at t=T vs Lambda')
plt.ylabel('Cumulative Returns')
plt.xlabel('Lambda')
plt.legend(loc='lower right')

# compare sharpe ratio
plt.figure(figsize=(10, 6))
plt.plot(lambda_list, SR_MVE, label="Mean-Variance Efficient Portfolio")
plt.plot(lambda_list, SR_regSw2, label="Regime Switching (2 regimes)")
plt.plot(lambda_list, SR_regSw4, label="Regime Switching (4 regimes)")
plt.title('Sharpe Ratio vs Lambda')
plt.ylabel('Sharpe Ratio')
plt.xlabel('Lambda')
plt.legend(loc='lower right')

# get values of Lambda that maximises portfolio cumulative returns and Sharpe Ratio
Lambda_max_MVE = lambda_list[portCumRet_MVE[:,254] == max(portCumRet_MVE[:,254])]
Lambda_max_MVE = lambda_list[SR_MVE == max(SR_MVE)]

Lambda_max_2 = lambda_list[portCumRet_regSw2[:,254] == max(portCumRet_regSw2[:,254])]
Lambda_max_2 = lambda_list[SR_regSw2 == max(SR_regSw2)]

Lambda_max_4 = lambda_list[portCumRet_regSw4[:,254] == max(portCumRet_regSw4[:,254])]
Lambda_max_4 = lambda_list[SR_regSw4 == max(SR_regSw4)]

# plot cumulative returns
plt.figure(figsize=(10, 6))
plt.plot(validation['date'], portCumRet_MVE[0,:], label="Lambda = 0.2 (max)")
plt.plot(validation['date'], portCumRet_MVE[4,:], label="Lambda = 1")
plt.plot(validation['date'], portCumRet_MVE[12,:], label="Lambda = 5")
plt.plot(validation['date'], portCumRet_MVE[14,:], label="Lambda = 7")
plt.plot(validation['date'], portCumRet_MVE[16,:], label="Lambda = 9")
plt.title('MVE Portfolio Cumulative Return')
plt.ylabel('Cumulative Returns')
plt.legend(loc='upper left')

plt.figure(figsize=(10, 6))
plt.plot(validation['date'], portCumRet_regSw2[0,:], label="Lambda = 0.2")
plt.plot(validation['date'], portCumRet_regSw2[4,:], label="Lambda = 1")
plt.plot(validation['date'], portCumRet_regSw2[12,:], label="Lambda = 5")
plt.plot(validation['date'], portCumRet_regSw2[14,:], label="Lambda = 7")
plt.plot(validation['date'], portCumRet_regSw2[16,:], label="Lambda = 9 (max)")
plt.title('Regime-Switching (2 Regimes) Portfolio Cumulative Return')
plt.ylabel('Cumulative Returns')
plt.legend(loc='upper left')

plt.figure(figsize=(10, 6))
plt.plot(validation['date'], portCumRet_regSw4[0,:], label="Lambda = 0.2")
plt.plot(validation['date'], portCumRet_regSw4[4,:], label="Lambda = 1")
plt.plot(validation['date'], portCumRet_regSw4[12,:], label="Lambda = 5 (max)")
plt.plot(validation['date'], portCumRet_regSw4[14,:], label="Lambda = 7")
plt.plot(validation['date'], portCumRet_regSw4[16,:], label="Lambda = 9")
plt.title('Regime-Switching (4 Regimes) Portfolio Cumulative Return')
plt.ylabel('Cumulative Returns')
plt.legend(loc='upper left')


# compute portfolio cumulative returns with Lambda_max
# without TC in MVO
w = MVE(Lambda_max_MVE)
port = portCumRetTC(w)
portCumRet_MVE = port[0]
   
w_regSwitch2 = regSw2(Lambda_max_2)
port2TC = portCumRetTC(w_regSwitch2)
portCumRet_regSw2 = port2TC[0]

w_regSwitch4 = regSw4(Lambda_max_4)
port4TC = portCumRetTC(w_regSwitch4)
portCumRet_regSw4 = port4TC[0]

# with TC in MVO
w = MVE_TC(Lambda_max_MVE)
port = portCumRetTC(w)
portCumRet_MVE = port[0]
   
w_regSwitch2_TC = regSw2_TC(Lambda_max_2)
port2TC = portCumRetTC(w_regSwitch2_TC)
portCumRet_regSw2 = port2TC[0]

w_regSwitch4_TC = regSw4_TC(Lambda_max_4)
port4TC = portCumRetTC(w_regSwitch4_TC)
portCumRet_regSw4 = port4TC[0]

# plot cumulative returns for all models
plt.figure(figsize=(10, 6))
plt.plot(validation['date'], portCumRet_MVE, label="Mean-Variance Efficient Portfolio after TC (Lambda = 5) ")
plt.plot(validation['date'], portCumRet_regSw2, label="Regime Switching (2 regimes) after TC (Lambda = 20)")
plt.plot(validation['date'], portCumRet_regSw4, label="Regime Switching (4 regimes) after TC (Lambda = 3.5)")
plt.ylabel('Cumulative Returns')
plt.legend(loc='upper left')


#----------------------------------------------------------------------------#
# Manual search of optimum lambda and tau values for Regime-Switching with   #
# Black-Litterman models                                                     #
#----------------------------------------------------------------------------#
# set list of lambda values to be tested
lambda_list = np.array([0.1, 0.2, 0.4, 0.6])
lambda_list = np.concatenate((lambda_list, np.arange(1, 6, 1)))
lambda_list = np.concatenate((lambda_list, np.arange(6, 11, 2)))

# set list of tau values to be tested
tau_list = np.arange(0.02, 0.1, 0.02)
tau_list = np.concatenate((tau_list, np.arange(0.1, 0.6, 0.1)))
tau_list = np.concatenate((tau_list, np.arange(0.6, 1.2, 0.2)))
#tau_list = np.concatenate((tau_list, np.array([2,3])))

portCumRet_regSw2BL = []
portCumRet_regSw4BL = []
SR_regSw2BL = np.zeros((len(lambda_list), len(tau_list)))
SR_regSw4BL = np.zeros((len(lambda_list), len(tau_list)))
cumRet_regSw2BL = np.zeros((len(lambda_list), len(tau_list)))
cumRet_regSw4BL = np.zeros((len(lambda_list), len(tau_list)))


# compute portfolio cumulative returns and Sharpe Ratio for each Lambda and tau values
# use function '....._TC()' for MVO algorithm with TC 
# functions without '_TC' indicates standard MVO algorithm
for i in range(0,len(lambda_list)):
    col1 = []
    col2 = []
    for j in range(0,len(tau_list)):
        optimVar=np.array([lambda_list[i], tau_list[j]])
        print(optimVar)
        
        w_regSwitch2_BL = regSw2BLabs(optimVar)
        port2TC = portCumRetTC(w_regSwitch2_BL)
        col1.append(port2TC[0])
        cumRet_regSw2BL[i][j] = port2TC[0][254]
        SR_regSw2BL[i][j] = port2TC[1]
    
        w_regSwitch4_BL = regSw4BLabs(optimVar)
        port4TC = portCumRetTC(w_regSwitch4_BL)
        col2.append(port4TC[0])
        cumRet_regSw4BL[i][j] = port4TC[0][254]
        SR_regSw4BL[i][j] = port4TC[1]
        
    portCumRet_regSw2BL.append(col1)
    portCumRet_regSw4BL.append(col2)


# plot results in heatmap
tau_list = np.array([0.02, 0.04, 0.06, 0.08, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1])

fig, ax = plt.subplots(figsize=(10, 8))
im, cbar = heatmap(cumRet_regSw4BL, lambda_list, tau_list, ax=ax,
                   cmap="YlGn", cbarlabel="Portfolio Cumulative Return")
texts = annotate_heatmap(im, valfmt="{x:.2f}")
fig.suptitle('Portfolio Cumulative Return for 4-Regime Model + Black-Litterman')
ax.set_ylabel('lambda')
ax.set_xlabel('tau')
plt.show()

fig, ax = plt.subplots(figsize=(10, 8))
im, cbar = heatmap(SR_regSw4BL, lambda_list, tau_list, ax=ax,
                   cmap="YlGn", cbarlabel="Sharpe Ratio")
texts = annotate_heatmap(im, valfmt="{x:.2f}")
fig.suptitle('Sharpe Ratio for 4-Regime Model + Black-Litterman')
ax.set_ylabel('lambda')
ax.set_xlabel('tau')
plt.show()

fig, ax = plt.subplots(figsize=(10, 8))
im, cbar = heatmap(cumRet_regSw2BL, lambda_list, tau_list, ax=ax,
                   cmap="YlGn", cbarlabel="Portfolio Cumulative Return")
texts = annotate_heatmap(im, valfmt="{x:.2f}")
fig.suptitle('Portfolio Cumulative Return for 2-Regime Model + Black-Litterman')
ax.set_ylabel('lambda')
ax.set_xlabel('tau')
plt.show()

fig, ax = plt.subplots(figsize=(10, 8))
im, cbar = heatmap(SR_regSw2BL, lambda_list, tau_list, ax=ax,
                   cmap="YlGn", cbarlabel="Sharpe Ratio")
texts = annotate_heatmap(im, valfmt="{x:.2f}")
fig.suptitle('Sharpe Ratio for 2-Regime Model + Black-Litterman')
ax.set_ylabel('lambda')
ax.set_xlabel('tau')
plt.show()


# compute portfolio cumulative returns with optimised Lambda and tau
w_MVE = MVE(5)
port = portCumRetTC(w_MVE)

w_porttest = regSw4BLabs(np.array([10, 0.02]))
porttest = portCumRetTC(w_porttest)

# plot graphs
plt.figure(figsize=(10, 6))
plt.plot(validation['date'], port[0], label='Mean-Variance Efficient Portfolio with Lambda=5')
plt.plot(validation['date'], porttest[0], 
         label='Regime Switching (4 regimes) + Black-Litterman model with Lambda=0.1, tau=0.2')
plt.ylabel('Cumulative Return')
plt.legend(loc='lower left')

# asset weights
fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(10, 6))
fig.suptitle('Asset Weights - Regime Switching (4 regimes) + Black-Litterman model with Lambda=0.1, tau=0.2')
ax1.plot(validation['date'], porttest[3]['JCI'], linewidth=1.0, label='JCI')
ax1.legend(loc='upper left')
ax2.plot(validation['date'], porttest[3]['Bond'], linewidth=1.0, label='Bond')
ax2.legend(loc='upper left')
ax3.plot(validation['date'], porttest[3]['cash'], linewidth=1.0, label='Risk Free')
ax3.legend(loc='upper left')


#----------------------------------------------------------------------------#
# Out-sample test                                                            #
#----------------------------------------------------------------------------#
# test for medium fees
bid = 0.45/100
ask = 0.55/100

# benchmark MVE with optimum lambda = 5
Lambda = 5
w = MVE_outSample(Lambda)
portMVE = portCumRetTC(w, test)

# out-sample test using regime switching (4 regime) model with TC in MVO 
# algorithm with optimum lambda = 7
Lambda = 7
w_outSample = regSw4_TC_outSample(Lambda)
portOutSample = portCumRetTC(w_outSample, test)

portOutSample_monthly = portCumRetTC_monthly(w_outSample, test)

# out-sample test using regime switching (2 regime) + Black-litterman model 
# with standard MVO algorithm, optimum lambda=1 and tau=0.04
Lambda = 1
tau = 0.04
w_outSample2 = regSw2BLabs_outSample(np.array([Lambda, tau]))
portOutSample2 = portCumRetTC(w_outSample2, test)

portOutSample2_monthly = portCumRetTC_monthly(w_outSample2, test)

JCIcumRet = np.cumprod(1+test['JCIWeeklyRet'])
bondcumRet = np.cumprod(1+test['BondWeeklyRet'])

# plot graphs
plt.figure(figsize=(10, 6))
plt.plot(test['date'], portMVE[0], label='Mean-Variance Efficient Portfolio with Lambda=5')
plt.plot(test['date'], portOutSample[0], 
         label='Regime Switching (4 regimes) with TC in MVO algorithm and Lambda=7')
plt.plot(test['date'], portOutSample_monthly[0], 
         label='Regime Switching (4 regimes) with TC in MVO algorithm and Lambda=7 (monthly rebalancing)')
plt.plot(test['date'], portOutSample2[0], 
         label='Regime Switching (2 regimes) + Black-Litterman with standard MVO algorithm, Lambda=1 and tau=0.04')
plt.plot(test['date'], portOutSample2_monthly[0], 
         label='Regime Switching (2 regimes) + Black-Litterman with standard MVO algorithm, Lambda=1 and tau=0.04  (monthly rebalancing)')
#plt.plot(test['date'], JCIcumRet, label='JCI')
#plt.plot(test['date'], bondcumRet, label='bond')
plt.ylabel('Cumulative Return')
plt.legend(loc='lower left', bbox_to_anchor=(-0.15, -0.35))

# asset weights for all models
fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(10, 6))
fig.suptitle('Asset Weights')
ax1.plot(test['date'], portMVE[3]['JCI'], linewidth=1.0, label='JCI - MVE')
ax1.plot(test['date'], portOutSample[3]['JCI'], linewidth=1.0, label='JCI - RegSw4')
ax1.plot(test['date'], portOutSample2[3]['JCI'], linewidth=1.0, label='JCI - RegSw2BL')
ax1.legend(loc='upper left')
ax2.plot(test['date'], portMVE[3]['Bond'], linewidth=1.0, label='Bond - MVE')
ax2.plot(test['date'], portOutSample[3]['Bond'], linewidth=1.0, label='Bond - RegSw4')
ax2.plot(test['date'], portOutSample2[3]['Bond'], linewidth=1.0, label='Bond - RegSw2BL')
ax2.legend(loc='upper left')
ax3.plot(test['date'], portMVE[3]['cash'], linewidth=1.0, label='Risk Free - MVE')
ax3.plot(test['date'], portOutSample[3]['cash'], linewidth=1.0, label='Risk Free - RegSw4')
ax3.plot(test['date'], portOutSample2[3]['cash'], linewidth=1.0, label='Risk Free - RegSw2BL')
ax3.legend(loc='upper left')
