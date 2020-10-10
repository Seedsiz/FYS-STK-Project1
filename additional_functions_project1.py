"""
Additional function used in Poly2DFit and project1_main
"""

from plotting_functions import plot_it, plot_stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from imageio import imread
import Poly2DFit

def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4


def MSE(data, model):
    """
    Calculates the Mean Squared Error if both data and model are vectos
    Calculates Variance if data is vector and model is the mean value of the data
    """
    n = np.shape(data)[0]
    res = np.array(data - model)
    return (1.0/n) *(res.T.dot(res))

def R2(data, model):
    """
    calculate the R2 score function
    """
    res = np.array(data - model)
    numerator = (res.T.dot(res))
    res1 = (data - np.mean(data))
    denominator = (res1.T.dot(res1))
    return 1 - numerator/denominator

def load_terrain(imname, sel = 4): #select every fourth
    """
    This function loads the terrain data. The data
    is then reduced by selecting every sel (eg. every 4th. element).
    It then flattens the reduced matrix and returns z(x,y) - height,
    and x,y pixel index.
    """
    terrain = imread('{:}.tif'.format(imname))
    # Show the terrain
    """
    plt.figure()
    plt.title(imname)
    plt.imshow(terrain, cmap='gray')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()
    """
    #reducing terrain data
    N = len(terrain[0,::sel]) # length reduced columns
    n = len(terrain[::sel,0]) # length reduced rows
    NN = len(terrain[0,:]) # number of columns total
    nn = len(terrain[:,0]) # number of rows total
    #reducing by column
    reduced  = np.zeros((nn,N))
    for i in range(nn):
            reduced[i,:] = terrain[i,::sel]
    #reduce by rows
    reduced2 = np.zeros((n,N))
    for j in range(N):
        reduced2[:,j] = reduced[::sel,j]
    #flattening
    z = reduced2.flatten()
    # creating arrays for x and y
    x_range = np.arange(1,n+1)
    y_range = np.arange(1,N+1)
    X,Y = np.meshgrid(x_range,y_range)
    x = X.flatten();y = Y.flatten()
    return x,y,z


def toi_append(data, info, regressiontype, lam, kFold):
    n = len(info['power'].to_numpy())
    app = pd.DataFrame(columns = ['Regression type','lambda','kFold',
                                        'Complexity','Value', 'Metric'] )
    for t in ['MSE', 'Bias', 'Variance', 'R2','MSE_train']:
        dat = np.array([[regressiontype for i in range(n)],
                       [lam for i in range(n)],
                       [kFold for i in range(n)],
                       info['power'].to_numpy(),
                       info[t.lower()].to_numpy(),
                       [t for i in range(n)]])
        temp = pd.DataFrame(dat.T, columns = ['Regression type','lambda','kFold',
                                        'Complexity','Value', 'Metric'] )

        app = app.append(temp)

    return  data.append(app)


def benchmarking( regressiontype, n = 500, order = 7, lam = 0.1, kfold = 0,
                 display_info = True, plot_info = True, plot_fit =False, save_file = False, x = 0, y = 0, z = 0):

    

    # initialuse the name of rows and columns
    ind = ['model_pow_%d'%i for i in np.arange(0,order)]
    col = ['power','mse', 'r2','bias', 'variance', 'mse_train']
    table_of_info = pd.DataFrame(index=ind, columns=col)
    
    # this is for naming the parameters in the table of parameters. commented out as not in use
    '''col1 = ['Regression type', 'lambda']
    for j in np.arange(0,order):
        for k in np.arange(0,j+1):
                name = 'x%s y%s'% (j-k,k)
                col1.append(name)    
    '''
    #Initialize a dataframe to store the results of coef_matrix:
    table_of_beta = pd.DataFrame(index=['beta', 'lower confidence level', 'upper confidence level'])
    
    #loop for creating fits for many orders
    for i in np.arange(0,order):
        #create fit object
        fit_object = Poly2DFit.Poly2DFit()

        #generate data with noise: mean 0, var =1
        fit_object.generateSample(n)

        # alternatively work with terrain data
        # fit_object.givenData(x,y,z)

        if kfold != 0:
            fit_object.kfold_cross(i, regressiontype, lam, kfold )
        else:
            #returns the fitted parameters and their variance
            par, par_var = fit_object.run_fit( i, regressiontype, lam )

            #evaluate model, return x,y points and model prediction
            fit_object.evaluate_model()

        
        #save informaton
        table_of_info.iloc[i,0] = i
        table_of_info.iloc[i,1] = fit_object.mse
        table_of_info.iloc[i,2] = fit_object.r2
        table_of_info.iloc[i,3] = fit_object.bias
        table_of_info.iloc[i,4] = fit_object.variance
        if kfold != 0:
            table_of_info.iloc[i,5] = fit_object.mse_train

        if plot_fit:
            #plot the data and model
            fit_object.plot_function()

        if save_file:
             #stores information about the fit in ./Test/OLS/first_OLS.txt
             fit_object.store_information(regressiontype, 'order_%i' % i)

        #calculate the 95% confidence interval
        conf_upper, conf_lower = confidence_int(fit_object.beta_mean, fit_object.variance, kfold)
        
        #find the number of beta parameters and then put these parameters in a table with the confidence interval
        num_of_para = int((i+1)*(i+2)/2)
        #loop through each parameter
        table_of_beta.loc[:,'regressiontype'] = regressiontype
        table_of_beta.loc[:,'lamda'] = lam
        for m in np.arange(0, num_of_para):
            table_of_beta.loc['beta',m] = fit_object.par[m]
            table_of_beta.loc['lower confidence level', m] = conf_lower[m]
            table_of_beta.loc['upper confidence level', m] = conf_upper[m]
            
        
    if display_info:
        pd.options.display.float_format = '{:,.2g}'.format
        print (table_of_info)
        print (table_of_beta)

    if plot_info:
        title = regressiontype +' Regression Info'
        if regressiontype != 'OLS':
            title += ' $\lambda$ = %.2f' % lam
        plot_stats(table_of_info,title )

    return table_of_info, table_of_beta


def confidence_int(mean, var, kfold):
    '''
    This is a function which uses the mean of the betas and the variance to calculate the 95% confidence interval
    The outputs are the upper and lower confidence levels. 
    '''
    conf_upper =  mean + (1.96 * np.sqrt(var / kfold))
    conf_lower =  mean - (1.96 * np.sqrt(var / kfold))
    
    return conf_upper, conf_lower
    