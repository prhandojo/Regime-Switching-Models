# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 12:06:07 2020

@author: prhandojo

library of functions for fitting regime-switching model: 
    
    genTheta0(data, m) 
        To generate array of initial model parameters theta_0 from in-sample
        data
        
        Parameters:
            data : in-sample data
            m : number of regimes
    
    is_pos_semi_def(x)
        To check if a matrix x is positive semi-definite or not
    
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
    
    logLik(theta, data, m)
        To return negative of log-likelihood, output from markovRegSwitch(), as an input to 
        the optimiser
        
    logLikGrad(theta, data, m)
        To compute gradient of function logLik() at value of theta
    

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy import optimize
from scipy.optimize import Bounds

# define functions 
# function to generate theta0 for 2 variables and m regimes
def genTheta0(data, m) :
    theta0 = []
    for i in range(0,m):
        # mean in each regime
        theta0[(0+5+(m-1))*i:(0+5+(m-1))*i+2] = np.mean(data)
        # variance and covariance in each regime
        vcov=np.cov(np.transpose(data))
        theta0[(0+5+(m-1))*i+2:(0+5+(m-1))*i+4] = np.diag(vcov)
        theta0.append(vcov[0][1])
        # regime probabilities
        for n in range(1,m):
            theta0.append(1/m)
    return np.array(theta0)  

# function to check if matrix is positive semidefinite
def is_pos_semi_def(x):
    return np.all(np.linalg.eigvals(x) >= 0)

# function to compute filtered regime probabilities and likelihood according to markov regime switching model
def markovRegSwitch(theta, data, m) :
    print('estimating regime probabilities..')
    print('parameters used:')
    print(theta)
    
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

# function to return negative log-likelihood, for input into optimiser
def logLik (theta, data, m) :
    return -markovRegSwitch(theta, data, m)[0]

# function to numerically compute gradient of logLik()
def logLikGrad (theta, data, m) :
    print('estimating gradient..')
    eps=theta*0.001
    return optimize.approx_fprime(theta, logLik, eps, data, m)


# import and format data
data_weekly = pd.read_csv("weeklyret.csv")
data_weekly['date'] = pd.to_datetime(data_weekly['date'],format='%m/%d/%Y')

# split into train, validation and test data set 
train = data_weekly.loc[data_weekly['year']<=2010,]
validation = data_weekly.loc[(data_weekly['year']>2010) & (data_weekly['year']<=2015),]
test = data_weekly.loc[data_weekly['year']>2015,]

# compute log returns 
data = pd.DataFrame()
data['JCILogRet'] = np.log(1+train['JCIExRet'])
data['bondLogRet'] = np.log(1+train['BondExRet'])

#------------------------------------------#
# estimate parameters for 2 regimes model  #
#------------------------------------------#
m = 2 # set number of regimes

# get 1st and 3rd quantile of the data for constraints on mean
JCIq1 = np.quantile(data['JCILogRet'], 0.25)
JCIq3 = np.quantile(data['JCILogRet'], 0.75)
bondq1 = np.quantile(data['bondLogRet'], 0.25)
bondq3 = np.quantile(data['bondLogRet'], 0.75)

bounds = Bounds([JCIq1, bondq1, 0.000001, 0.000001, -np.inf, 0, JCIq1, bondq1, 0.000001, 0.000001, -np.inf, 0], 
                [JCIq3, bondq3, np.inf, np.inf, np.inf, 1, JCIq3, bondq3, np.inf, np.inf, np.inf, 1])

# generate theta0 
theta0 = genTheta0(data, m)

# run optimiser to estimate parameters by maximising likelihood
result = optimize.minimize(fun = logLik, 
                            x0 = theta0, 
                            args = (data, m),
                            method = 'trust-constr', 
                            jac = logLikGrad,
                            bounds = bounds,
                            options = {'disp': True})
theta22 = result.x
np.savetxt("theta22.csv", theta22)

# import previously ran parameters
theta22 = np.array(pd.read_csv("2var2regime_id_weekly_log.csv")['x'])

# use the parameters to estimate filtered regime probabilities
Filter = markovRegSwitch(theta22, data, m)[1]
Filter['t index'] = list(range(0, len(data)))
Filter['Regime 1'] = Filter['p1']>0.5
Filter['Regime 2'] = Filter['p2']>0.5

xx = Filter[Filter['Regime 1']==True]['t index']

# plot graphs
fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(12, 6))
fig.suptitle('JCI & bond log returns and filtered probability for regime 1')
ax1.plot(Filter['t index'], data['JCILogRet'], linewidth=1.0)
ax2.plot(Filter['t index'], data['bondLogRet'], linewidth=1.0)
ax3.plot(Filter['t index'], Filter['p1'], linewidth=1.0)


#------------------------------------------#
# estimate parameters for 4 regimes model  #
#------------------------------------------#
m = 4 # set number of regimes

# get 1st and 3rd quantile of the data for constraints on mean
JCIq1 = np.quantile(data['JCILogRet'], 0.25)
JCIq3 = np.quantile(data['JCILogRet'], 0.75)
bondq1 = np.quantile(data['bondLogRet'], 0.25)
bondq3 = np.quantile(data['bondLogRet'], 0.75)

bounds = Bounds([JCIq1, bondq1, 0.000001, 0.000001, -np.inf, 0, 0, 0, 
                 JCIq1, bondq1, 0.000001, 0.000001, -np.inf, 0, 0, 0,
                 JCIq1, bondq1, 0.000001, 0.000001, -np.inf, 0, 0, 0,
                 JCIq1, bondq1, 0.000001, 0.000001, -np.inf, 0, 0, 0], 
                [JCIq3, bondq3, np.inf, np.inf, np.inf, 1, 1, 1, 
                 JCIq3, bondq3, np.inf, np.inf, np.inf, 1, 1, 1,
                 JCIq3, bondq3, np.inf, np.inf, np.inf, 1, 1, 1,
                 JCIq3, bondq3, np.inf, np.inf, np.inf, 1, 1, 1])

# generate theta0 
theta0 = genTheta0(data, m)

# run optimiser to estimate parameters by maximising likelihood
result = optimize.minimize(fun = logLik, 
                            x0 = theta0, 
                            args = (data, m),
                            method = 'trust-constr', 
                            jac = logLikGrad,
                            bounds = bounds,
                            options = {'disp': True})
theta24 = result.x
np.savetxt("theta24.csv", theta24)

# import previously ran parameters
theta24 = np.array(pd.read_csv("2var4regime_id_weekly_log.csv")['x'])

# use the parameters to estimate filtered regime probabilities
Filter = markovRegSwitch(theta24, data, m)[1]
Filter['t index'] = list(range(0, len(data)))
Filter['Regime 1'] = Filter['p1']>0.5
Filter['Regime 2'] = Filter['p2']>0.5
Filter['Regime 3'] = Filter['p3']>0.5
Filter['Regime 4'] = Filter['p4']>0.5

xx = Filter[Filter['Regime 1']==True]['t index']

# plot graphs
fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(12, 6))
fig.suptitle('JCI & bond log returns and filtered probability for regime 1')
ax1.plot(Filter['t index'], data['JCILogRet'], linewidth=1.0)
ax2.plot(Filter['t index'], data['bondLogRet'], linewidth=1.0)
ax3.plot(Filter['t index'], Filter['p1'], linewidth=1.0)

