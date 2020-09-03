#!/usr/bin/env python
# coding: utf-8

# In[1]:


#recreates homework #1 from econ 534 in Python
#test for heteroskedasticity using the White (1980) test
#also show how to use WLS/GLS to correct for hetereoskedasticity
#White test for heteroskedasticity
#call in homepriceV10.dta data


# In[2]:


import pandas as pd
hpv10data = 'C:/Users/jfras/OneDrive/UND/534AppliedEcon/Datasets/homepriceV10.dta'
#make sure you preserve data types, otherwise it all comes in as strings
df = pd.read_stata(hpv10data, preserve_dtypes=True)
df.tail()


# In[3]:


print('Check your data Datatypes after import')
print(df.dtypes)


# In[4]:


df.describe()


# In[5]:


df.shape
#(rows,columns)


# In[42]:


#have to execute this block twice before graph shows up 7-14-2020
import numpy as np
import matplotlib.pyplot as plt
#some options...
#plt.ion()
#plt.figure(figsize=(15,10))
plt.tight_layout()
#now the plot
df.plot(x='floor', y='price', style='o')
plt.title('Square Feet vs Price')
plt.xlabel('Floor')  
plt.ylabel('Price')
plt.pause(.10)
plt.show(block=False)

#legend disappears with this method below
##plt.plot('floor', 'price', 'o', data=df)


# In[7]:


import seaborn as seabornInstance 
#plt.figure(figsize=(15,10))
#plt.tight_layout()
seabornInstance.distplot(df['price'])


# In[8]:


#setup 3 variables as independent variables
# tried df2 = pd.DataFrame(hpv10data,columns=['id','price','floor','lot','bath','bed','bathbed','year','age','agesq','gar','status','dac','school','ded','dha','dad','dcr','dpa'])
# tried X=df['floor'].values.reshape(-1,1)
X = pd.read_stata(hpv10data,columns=['floor', 'lot', 'bed']).values
X.shape


# In[9]:


y = df['price'].values.reshape(-1,1)
y.shape


# In[10]:


#create the regression using stats models
#could have done it this way but
#sklearn does not have nice anova table so we will use stats models
#from sklearn.linear_model import LinearRegression
#regress price on lot, floor, bed
#regr = LinearRegression()
#regr.fit(X, y)

#using stats models
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
#regress price on lot, floor, bed
expr1 = 'y ~ lot + floor + bed'
lm1 = ols(expr1, df).fit()
print(lm1.summary())

#print(regr.intercept_)
#print(regr.coef_)


# In[11]:


#you can use these commands to print the attributes of the regeression, similair to stata's "e"
#print("summary()\n",lm1.summary())
#print("pvalues()\n",lm1.pvalues)
#print("tvalues()\n",lm1.tvalues)
print("rsquared()\n",lm1.rsquared)
#print("rsquared_adj()\n",lm2.rsquared_adj)
print("parameters()\n",lm1.params)


# In[12]:


#now show the residual, white noise?


# In[13]:


#first create the predicted value of y
yhat = lm1.predict()
print(yhat)
plt.plot(X, y, 'o', color='black')
plt.xlabel('Lot Size')  
plt.ylabel('Price')
plt.title("Actuals vs Regresion Line")

#https://www.statsmodels.org/v0.10.2/examples/notebooks/generated/predict.html


# In[ ]:





# In[14]:


# get residual (uhat)
uhat= y-yhat
plt.plot(X,uhat, 'o', color='darkblue')
plt.title("Residual Plot")
plt.xlabel("Independent Variable")
plt.ylabel("Residual")


# In[15]:


#some white test instructions
#http://hamelg.blogspot.com/2015/11/python-for-data-analysis-part-25-chi.html
#https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chisquare.html
#https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.power_divergence.html#scipy.stats.power_divergence
#https://www.programiz.com/python-programming/tuple
#https://mgimond.github.io/Stats-in-R/ChiSquare_test.html


# In[16]:


#SHORT METHOD FOR WHITE TEST for expr1


# In[17]:


# now finally the white test for heteroskedasticity, using patsy dmatrices function from statsmodels library
from statsmodels.stats.diagnostic import het_white
from statsmodels.compat import lzip
from patsy import dmatrices
#prep the expression for the dmatrices function
#from above expr1 = 'y ~ lot + floor + bed'
#set up the dataframe
uhat, X = dmatrices(expr1, df, return_type='dataframe')
#execute the test
keys = ['Lagrange Multiplier statistic:', 'LM test\'s p-value:', 'F-statistic:', 'F-test\'s p-value:']
results = het_white(uhat, X)
lzip(keys, results)
#print("heteroskedasticity is a problem here")


# In[18]:


#2ND SHORT METHOD OF WHITE TEST FOR expr1
#white test
#statsmodels.stats.diagnostic.het_white(residuals-array, regressors-array)
sm.stats.diagnostic.het_white(uhat, X)

#https://medium.com/keita-starts-data-science/heteroskedasticity-in-linear-regressions-and-python-16eb57eaa09


# In[19]:


#print White's 1980 Robust Standard errors
print("HC0_se()\n",lm1.HC0_se)
#https://www.statsmodels.org/0.8.0/generated/statsmodels.regression.linear_model.RegressionResults.html#statsmodels.regression.linear_model.RegressionResults


# In[ ]:





# In[20]:


#LONG METHOD FOR WHITE TEST
#now do the white test for heteroskedasticity showing the work done by the library function het_white behind the scenes


# In[21]:


#create some interaction variables
df['uhat_2'] = uhat * uhat
df['lot_floor'] = df.lot * df.floor
df['lot_bed'] = df.lot * df.bed
df['floor_bed'] = df.floor * df.bed
df['lot_2'] = df.lot * df.lot
df['floor_2'] = df.floor * df.floor
df['bed_2'] = df.bed * df.bed
#check the dataframe, previously was 19 columns, should have added 7 columns for a total of 26
df.shape
#df.tail()


# In[22]:


#examine the 7 new columns
df.tail()
#if you make a mistake you can drop a variable with for example df.drop([uhat_2,axis=1])
#axis=1 means column not a row


# In[23]:


#using statsmodels
#and using these interactive variables and the residual uhat from above
#regress residual squared on 'floor', 'lot', 'bed','lot_floor','lot_bed','floor_bed','lot_2','floor_2','bed_2'
expr2 = 'uhat_2 ~ lot + floor + bed + lot_floor + lot_bed + floor_bed + lot_2 + floor_2 + bed_2'
lm2 = ols(expr2, df).fit()
print(lm2.summary())


# In[24]:


#you can use these commands to print the attributes of the regeression, similair to stata's "e"
#print("summary()\n",lm2.summary())
#print("pvalues()\n",lm2.pvalues)
#print("tvalues()\n",lm2.tvalues)
print("rsquared()\n",lm2.rsquared)
#print("rsquared_adj()\n",lm2.rsquared_adj)
print("parameters()\n",lm2.params)


# In[25]:


#now declare the rsquared attribute for later use
rsquaredlm2 = lm2.rsquared
print (rsquaredlm2)
#then delcare degrees of freedom for the model attribute for later use
degfreedmodlm2 = lm2.df_model
print(degfreedmodlm2)
#then declare degrees of freedom for the residutal attribute for later use
degfreedreslm2 = lm2.df_resid
print(degfreedreslm2)
#then declare number of observations for regression attribute for later use
numobslm2 = lm2.nobs
print(numobslm2)


# In[ ]:





# In[26]:


#an excercise in confirming the X2 significance level tables
from scipy.stats import chi2
value = chi2.ppf(0.95, degfreedmodlm2)
print(value)

#confirm with cdf
p= chi2.cdf(value, degfreedmodlm2)
print(p)


# In[27]:


#THIS IS THE FINAL STEP IN THE LONG METHOD FOR THE WHITE TEST, (NOT MATCHING STATA RESULTS YET 7-14-2020)


# In[28]:


#You can carry out a chi-squared goodness-of-fit test automatically using the scipy function scipy.stats.chisquare():
#stats.chisquare(f_obs= observed,   # Array of observed counts
#                f_exp= expected)   # Array of expected counts
from scipy.stats import chisquare
obs=df['price'].values.reshape(-1,1)
obs.shape
chisquare(f_obs= obs, f_exp=285.79611)
#gives you this result
#Power_divergenceResult(statistic=array([955.25186468]), pvalue=array([1.09062857e-152]))
#chisquare(f_obs= obs)
#gives you this result, 
#Power_divergenceResult(statistic=array([955.2521], dtype=float32), pvalue=array([1.09052254e-152]))


# In[29]:


#POWER DIVERGENT RESULT
from scipy.stats import power_divergence
obs.shape
power_divergence(obs, axis=None)


# In[30]:


####for further research####
#https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.PredictionResults.html#statsmodels.regression.linear_model.PredictionResults


# In[31]:


#1st try, check regrssion using robust standard errors, se's doesn't match stata lesson 7-15-2020
#using statsmodels
#and using these interactive variables and the residual uhat from above
expr3 = 'price ~ lot + floor + bed'
lm3 = ols(expr3, df).fit()
print(lm3.summary())

#resources
#https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLSResults.get_robustcov_results.html
#https://www.statsmodels.org/0.8.0/rlm.html?highlight=robust%20linear%20model#module-statsmodels.robust


# In[32]:


#print White's 1980 Robust Standard errors
print("HC0_se()\n",lm3.HC0_se)
#https://www.statsmodels.org/0.8.0/generated/statsmodels.regression.linear_model.RegressionResults.html#statsmodels.regression.linear_model.RegressionResults


# In[34]:


#using statsmodels robust linear model
#lm4.get_robustcov_results(cov_type='HC1', use_t=None, **kwargs)
y4 = df['price'].values.reshape(-1,1)
X4 = pd.read_stata(hpv10data,columns=['floor', 'lot', 'bed']).values
rlm_model4=sm.RLM(y4, X4, M=sm.robust.norms.HuberT())
rlm_results4=rlm_model4.fit()
print(rlm_results4.params)
print(rlm_results4.summary())
# The line below is not defined
#lm4.get_robustcov_results(cov_type='HC1', use_t=None)


# In[36]:


#using robust linear model - rlm but this time we'll add a constant
y5 = df['price'].values.reshape(-1,1)
X5 = pd.read_stata(hpv10data,columns=['floor', 'lot', 'bed']).values
X5 = sm.add_constant(X5)
#rlm_model5=sm.RLM(y5, X5, M=sm.robust.norms.HuberT())
rlm_model5=sm.RLM(y5, X5, M=sm.robust.norms.HuberT())
rlm_results5=rlm_model5.fit()
print(rlm_results5.params)
print(rlm_results5.summary())
#https://www.statsmodels.org/devel/rlm.html


# In[37]:


print(rlm_results5.params)


# In[38]:


#second try, still don't match stata lesson se's 7-15-2020
##check regrssion using robust standard errors
expr6 = 'price ~ lot + floor + bed'
lm6 = ols(expr6, df).fit()
#lm6.get_robustcov_results(cov_type='HC1', use_t=None, **kwargs)
lm6.get_robustcov_results(cov_type='HC1', use_t=None)
print(lm6.summary())
#print aic for an example of attribute syntax, \n means new line
print("aic()\n",lm6.aic)

#resources
#https://www.statsmodels.org/stable/rlm.html
#https://www.statsmodels.org/stable/examples/notebooks/generated/robust_models_0.html
#https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.RegressionResults.html
#class statsmodels.regression.linear_model.RegressionResults
    #(model, params, normalized_cov_params=None, scale=1.0, cov_type='nonrobust', cov_kwds=None, use_t=None, **kwargs)[source]
#https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.RegressionResults.html


# In[39]:


print(lm6.params)


# In[40]:


#print White's 1980 Robust Standard errors
print("HC0_se()\n",lm6.HC0_se)
#https://www.statsmodels.org/0.8.0/generated/statsmodels.regression.linear_model.RegressionResults.html#statsmodels.regression.linear_model.RegressionResults


# In[ ]:





# In[41]:


#run regression with rlm, robust linear model from statsmodels
#rlm does not produce the same attributes as statsmodels ols
from statsmodels.formula.api import ols, rlm
rlm_model7= rlm('price ~ lot + floor + bed', df).fit()
print(rlm_model7.summary())
#can not print White's 1980 Robust Standard errors, rlm does not produce
#print("cov_HC3_se()\n",rlm_model.HC7_se)

#resources
#https://www.statsmodels.org/stable/examples/notebooks/generated/robust_models_1.html
#https://www.statsmodels.org/0.8.0/generated/statsmodels.regression.linear_model.RegressionResults.html#statsmodels.regression.linear_model.RegressionResults


# In[ ]:




