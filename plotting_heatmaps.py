# -*- coding: utf-8 -*-
"""
The following code imports the benchmarking csv from the Results, 
processes the data, 
and outputs heatmaps for the MSE as a function of lambda and complexity.

The code also plots test train splits for the optimal lambda. the optimal lambda must be manually entered.
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#%matplotlib inline
from plotting_functions import plotting_mse

# Read data from file 'filename.csv' 
# (in the same directory that your python process is based) 
data_load = pd.read_csv("Results/benchmarking.csv") 
 
# filter the necessary data
ridge_filter = ((data_load['Regression type'] == 'RIDGE') & (data_load['Metric'] != 'R2') & (data_load['Metric'] != 'Bias') & (data_load['Metric'] != 'Variance') & (data_load['Metric'] != 'MSE_train'))
lasso_filter = ((data_load['Regression type'] == 'LASSO') & (data_load['Metric'] != 'R2') & (data_load['Metric'] != 'Bias') & (data_load['Metric'] != 'Variance') & (data_load['Metric'] != 'MSE_train'))
ols_filter = ((data_load['Regression type'] == 'OLS') & (data_load['Metric'] != 'R2') & (data_load['Metric'] != 'Bias') & (data_load['Metric'] != 'Variance') & (data_load['Metric'] != 'MSE_train'))


##################  #ridge heatmap #############################################
f, ax = plt.subplots(figsize=(9, 6))

df_RIDGE = pd.pivot_table(data = data_load[ridge_filter],
                    values = 'Value',
                    index = 'Complexity',
                    columns = 'lambda')

g = sns.heatmap(df_RIDGE,annot=True, fmt=".3f")
plt.savefig(fname ='ridge_heatmap', dpi='figure', format= 'pdf')


#######################lasso heatmap #########################################
f1, ax1 = plt.subplots(figsize=(9, 6))

df_LASSO = pd.pivot_table(data = data_load[lasso_filter],
                    values = 'Value',
                    index = 'Complexity',
                    columns = 'lambda')

#ax.ticklabel_format(style='scientific')

#plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
#         rotation_mode="anchor")

g1=sns.heatmap(df_LASSO,annot=True, fmt=".3f", ax = ax1)
plt.savefig(fname ='lasso_heatmap', dpi='figure', format= 'pdf')

########################## OLS heatmap #####################################
f2, ax2 = plt.subplots(figsize=(6, 6))

df_OLS = pd.pivot_table(data = data_load[ols_filter],
                    values = 'Value',
                    index = 'Complexity',
                    columns = 'lambda')

g = sns.heatmap(df_OLS,annot=True, fmt=".3f", ax = ax2)
plt.savefig(fname ='ols_heatmap', dpi='figure', format= 'pdf')

#close all figures
plt.close('all')

################ test train plots for optimal lambda ######################

testrain_ridge_filter = ((data_load['Regression type'] == 'RIDGE') & (data_load['lambda'] == 1.0) & (data_load['Metric']=='MSE') | (data_load['Regression type'] == 'RIDGE') & (data_load['lambda'] == 1.0) & (data_load['Metric']=='MSE_train'))
testrain_lasso_filter = ((data_load['Regression type'] == 'LASSO') & (data_load['lambda'] == 0.01) & (data_load['Metric']=='MSE') | (data_load['Regression type'] == 'LASSO') & (data_load['lambda'] == 0.01) & (data_load['Metric']=='MSE_train'))
testrain_ols_filter =  ((data_load['Regression type'] == 'OLS') & (data_load['Metric']=='MSE') | (data_load['Metric']=='MSE_train') & (data_load['lambda'] == 0))

#data_load[testrain_ols_filter].to_csv('testrainridge.csv')

ylabel = 'MSE'
max_order = int(data_load['Complexity'].max())
g = sns. FacetGrid(data_load[testrain_ridge_filter], row = 'Regression type', col='kFold', hue ='Metric', margin_titles =True)
g1 = sns. FacetGrid(data_load[testrain_lasso_filter], row = 'Regression type', col='kFold', hue ='Metric', margin_titles =True)
g2 = sns. FacetGrid(data_load[testrain_ols_filter], row = 'Regression type', col='kFold', hue ='Metric', margin_titles =True)

#plt.ylim(0.8,1.3)
g.map(plt.plot, 'Complexity', 'Value')
g.add_legend()
g.set_axis_labels('Polynom Order', ylabel)
g.set(xticks =np.linspace(0,max_order,5,dtype=int))
g.savefig( 'testrainridge.pdf')

#plt.ylim(0.8,1.3)
g1.map(plt.plot, 'Complexity', 'Value')
g1.add_legend()
g1.set_axis_labels('Polynom Order', ylabel)
g1.set(xticks =np.linspace(0,max_order,5,dtype=int))
g1.savefig( 'testrainlasso.pdf')

#plt.ylim(0.8,1.3)
g2.map(plt.plot, 'Complexity', 'Value')
g2.add_legend()
g2.set_axis_labels('Polynom Order', ylabel)
g2.set(xticks =np.linspace(0,max_order,5,dtype=int))
g2.savefig( 'testrainols.pdf')

    
#show plots
plt.show