"""
functions for plotting the table of informations toi
and other plotting functions
"""
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

def duplicate_axis(ax):
    l = ax.lines[0]
    iter_line = True
    it = 0
    x = l.get_xdata()
    y = l.get_ydata()
    it += 1
    while iter_line:
        try:
            l = ax.lines[it]
            xt = l.get_xdata()
            yt = l.get_ydata()
            it += 1
            x = np.vstack([x,xt])
            y = np.vstack([y,yt])
        except:
            iter_line = False
    return x, y

def plotting_mse(toi, row, col, filename, split = False, ylabel ='MSE', shary = True):
    """
    gives the table of informations, toi, to facetgrid (apply filter in call)
    spans dimension by col, row with toi column names
    saves as filename
    if split is True: store each subplot seperatly
    if shary is True: shared y axes (only in split)
    """
    max_order = int(toi['Complexity'].max())
    g = sns. FacetGrid(toi, row =row, col=col, hue ='Metric', margin_titles =True)
    g.map(plt.plot, 'Complexity', 'Value')
    g.add_legend()
    g.set_axis_labels('Polynom Order', ylabel)
    g.set(xticks =np.linspace(0,max_order,5,dtype=int))
    g.savefig(filename + '.pdf')

    if split:
        labels = toi['Metric'].unique()
        axes = g.axes.flat
        for i,ax in enumerate(axes):
            title =ax.get_title()
            xax = ax.get_xlim()
            yax =ax.get_ylim()
            x,y  = duplicate_axis(ax)

            f = plt.figure(figsize=(10,10))
            for k in range(len(x)):
                plt.plot(x[k], y[k], label = labels[k])
            plt.xlim(xax)
            if shary:
                plt.ylim(yax)
            plt.ylabel(ylabel, fontsize = 24)
            plt.xlabel('Polynom Order', fontsize = 24)
            plt.legend(loc='best' , fontsize = 24)
            plt.savefig(fname=filename +str(i)+'_kf' +title[-2:], dpi='figure', format= 'pdf')
    plt.close('all')

def plotting_r2(toi, filename):
    max_order = int(toi['Complexity'].max())
    g = sns. FacetGrid(toi, col ='lambda', row='kFold', hue ='Regression type', margin_titles =True)
    g.map(plt.plot, 'Complexity', 'Value')
    g.set_axis_labels('Polynom Order', 'R2')
    g.add_legend()
    g.set(xticks = np.linspace(0,max_order,5,dtype=int))
    g.savefig(filename +'.pdf')
    labels = toi['Metric'].unique()
    axes = g.axes.flat
    labels = ['RIDGE', 'LASSO']
    for i, ax in enumerate(axes):
        xax = ax.get_xlim()
        x,y  = duplicate_axis(ax)
        plt.figure(figsize=(10,10))
        if x.ndim == 1:
            plt.plot(x,y)
        else:
            for k in range(len(x)):

                plt.plot(x[k], y[k], label = labels[k])
            plt.legend(loc='best' , fontsize = 24)

        plt.xlim(xax)
        plt.ylabel('R2', fontsize = 24)
        plt.xlabel('Polynom Order', fontsize = 24)
        plt.savefig(fname =filename +str(i), dpi='figure', format= 'pdf')
        plt.close('all')

def plotting(toi, folder = ''):
     #filter for lam
    lam_filter =  ((toi['lambda'] == 0) | (toi['lambda'] == 0.00001) | (toi['lambda'] == 0.001)) & ((toi['Metric'] != 'R2') & (toi['Metric'] != 'MSE_train'))
    #filter for Ridge
    ridge_filter = (toi['Regression type'] == 'RIDGE') & ((toi['Metric'] != 'R2') & (toi['Metric'] != 'MSE_train'))
    #filter for lasso
    lasso_filter = (toi['Regression type'] == 'LASSO') & ((toi['Metric'] != 'R2') & (toi['Metric'] != 'MSE_train'))
    #R2 filter
    r2_filter = toi['Metric'] =='R2'
    #book filter
    book_filter = ((toi['Metric']=='MSE') | (toi['Metric']=='MSE_train')) &  ((toi['lambda'] == 0) | (toi['lambda'] == 0.00001)) & (toi['kFold'] != 0)

    #compare kfold for different regressions
    plotting_mse(toi[lam_filter], col ='Regression type', row='kFold', filename = folder +'reg_types')
    #compare kfold and lambda for lasso
    plotting_mse(toi[lasso_filter], col ='lambda', row='kFold', filename = folder +'lasso_lam_vs_kfold')
    #compare kfold and lambda fore ridge
    plotting_mse(toi[ridge_filter], col ='lambda', row='kFold', filename = folder + 'ridge_lam_vs_kfold')
    #make r2 plot and split (no shared y)
    plotting_r2(toi[r2_filter], filename = folder +'r2')
    #make plot from book
    plotting_mse(toi[book_filter], col='Regression type', row='kFold', filename=folder +'train_vs_test', split = True, shary=True)
    plotting_mse(toi[book_filter], col ='Regression type', row='kFold', filename=folder +'train_vs_test_no_share', split = True, shary=False)
    

def plot_stats(info, title = 'Regression Infos'):
    data = pd.DataFrame(columns = ['Complexity','Value', 'Metric'] )
    n = len(info['power'].to_numpy())
    for t in ['MSE', 'Bias', 'Variance']:
        dat = np.array( [info['power'].to_numpy(),  info[t.lower()].to_numpy(), [t for i in range(n)]])

        app = pd.DataFrame(dat.T, columns =['Complexity','Value', 'Metric'] )
        data = data.append(app)

    plt.title(title)
    sns.lineplot(x = 'Complexity', y= 'Value', hue = 'Metric', data = data, estimator = None)
    plt.ylim(0, 1.5 ) #for plot comparison
    plt.show()

def plot_it(x,y,model,franke_data):
    '''
    This is a function to plot the x y and z data
    Inputs: x: the generated x points
            y: the generated y points
            model: the model we are testing
            franke_data : the data from the frankefunction
    '''
    ax = plt.axes(projection='3d')

    # plots scatter and trisurf
    ax.plot_trisurf(x, y, model, cmap='viridis', edgecolor='none')
    ax.scatter(x,y,franke_data)

    #set the axis labels and title
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('The sample data plotted as a scatter & the model plotted as a trisurf')

    plt.show()
    return ax
