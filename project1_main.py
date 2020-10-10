import Poly2DFit
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from additional_functions_project1 import benchmarking, toi_append, load_terrain
from plotting_functions import plotting
from imageio import imread
import time
def normalize(x,y,z, rescale = True):
    """
    normalize x,y, z 
    if rescale = True -> shift mean(z) -> 0
    """
    x = x / x.max()
    y = y / y.max()
    z =  z / z.max()
    if rescale:
        z -= np.mean(z)
    return x,y,z

def main():
    
    ks = [5]
    lam = [ 10**(-3), 10**(-1)]

    max_order = 3
    
    samples = 5*10**2

    toi = pd.DataFrame(columns = ['Regression type','lambda','kFold',
                                        'Complexity','Value', 'Metric'] )
    
    tob = pd.DataFrame()
    
    ops = len(ks) * ( 2 * len(lam) + 1)
    one_part = 100 / ops #in%
    current_progress = 0

# load terrain data, for example yellowstone
#    x,y,z = load_terrain('yellowstone')
#Give x, y,z as variables to benchmarking when working with terrain data.

    for k in ks:
        current_progress += one_part
        print("Now: OLS; Progress", int(current_progress), "%")
        temp, temp_tob = benchmarking('OLS', samples, max_order+1, kfold=k, plot_info= False, display_info = False)
        toi = toi_append(toi, temp, 'OLS', 0, k)
        tob = tob.append(temp_tob)
        
        for l in lam:
            current_progress += one_part
            print("Now: RIDGE; Progress", int(current_progress), "%")
            temp, temp_tob = benchmarking('RIDGE', samples, max_order+1, lam=l, kfold= k, plot_info= False, display_info = False)
            toi = toi_append(toi, temp, 'RIDGE', l, k)
            tob = tob.append(temp_tob)

            current_progress += one_part
            print("Now: LASSO; Progress", int(current_progress), "%")
            temp, temp_tob = benchmarking('LASSO', samples, max_order+1, lam=l, kfold= k, plot_info= False, display_info = False)
            toi = toi_append(toi, temp, 'LASSO', l, k)
            tob = tob.append(temp_tob)
            
    # plot results of benchmarking
    toi.to_csv('./Results/benchmarking.csv')
    tob.to_csv('./Results/beta.csv')
    plotting(toi, folder='')
    
    reductions = [6, 36]
    ks = [5]
    lam = [10**(-6), 10**(-5), 10**(-4), 10**(-3)]
    max_order = [ np.linspace(1,25,10,  dtype =int), np.arange(10,111,10), range(1,11)] 
    for reds, reduction in enumerate(reductions):
        
        x,y,z = load_terrain('./Terraindata/yellowstone1', reduction)
        x,y,z = normalize(x,y,z)
        
        fit_terain = Poly2DFit.Poly2DFit()
        fit_terain.givenData(x,y,z)
        toi = pd.DataFrame(columns = ['Regression type','lambda','kFold',
                                            'Complexity','Value', 'Metric'])

        ops = len(ks) * ( 2 * len(lam) + 1)*len(max_order[reds])
        one_part = 100 / ops #in%
        current_progress = 0
        
        for k in ks:
            for p in max_order[reds]:
                current_progress += one_part
                print("Now: OLS; Progress", int(current_progress), "%")
                try:
                    if k == 0:
                        fit_terain.run_fit(p, 'OLS')
                        fit_terain.evaluate_model()
                    else:
                        fit_terain.kfold_cross(p, 'OLS', k = k)
                    
                    temp = pd.DataFrame(np.array([[p,fit_terain.mse, fit_terain.r2,
                                        fit_terain.bias, fit_terain.variance, fit_terain.mse_train]], dtype=np.float),
                                        columns =['power','mse', 'r2','bias', 'variance', 'mse_train'])
                    
                    toi = toi_append(toi, temp, 'OLS', 0, k)
                except:
                    print('Failed OLS at order %i'%p)

                for l in lam:
                    for reg in ['RIDGE', 'LASSO']:
                        current_progress += one_part
                        print("Now: %s; Progress"% reg, int(current_progress),"%" )
                        try:
                            if k == 0:
                                fit_terain.run_fit(p, reg, l)
                                fit_terain.evaluate_model()
                            else:
                                fit_terain.kfold_cross(p, reg, k = k)
                        
                            temp = pd.DataFrame(np.array([[p,fit_terain.mse, fit_terain.r2,
                                                fit_terain.bias, fit_terain.variance, fit_terain.mse_train]], dtype=np.float),
                                                columns =['power','mse', 'r2','bias', 'variance', 'mse_train'])
                        
                            toi = toi_append(toi, temp, reg, l, k)
                        except:
                            print('Failed %s at order %i'%(reg,p))
        

        # plot results of fit
        toi['Value'] = toi['Value'].astype(float)
        toi['Complexity'] = toi['Complexity'].astype(float)
        toi['lambda'] = toi['lambda'].astype(float)
        toi.to_csv('./yellowstone1_%i_scale/benchmarking.csv' % reduction)
        plotting(toi, folder='./yellowstone1_%i_scale/'% reduction)
    
    SMALL_SIZE = 22
    MEDIUM_SIZE = 24
    BIGGER_SIZE = 28

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=0)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=0)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=0)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    
    reductions = [6, 36]
    for red in reductions:
        toi = pd.read_csv('./yellowstone1_%i_scale/benchmarking.csv'% red)

        x,y,z = load_terrain('./Terraindata/yellowstone1', red)
        x,y,z = normalize(x,y,z)
        fit_best = Poly2DFit.Poly2DFit()
        fit_best.givenData(x,y,z)

        plt.figure(figsize=(10,10))
        terrain = imread('./Terraindata/yellowstone1.tif')
        c1 = plt.imshow(terrain, cmap='gist_earth')
        plt.colorbar(c1,label='Z')
        plt.savefig('./yellowstone1_%i_scale/raw.pdf'% red)    
        
        zl = terrain[::red,::red].shape
        terrain = z.reshape(zl)
        tmin, tmax = terrain.min(), terrain.max()
    
        plt.figure(figsize=(10,10))
        plt.subplot(221)
        plt.title('Processed Data')
        c = plt.imshow(terrain, cmap='gist_earth', vmin=0.9*tmin, vmax=1.1*tmax)

        for i,reg in enumerate(['OLS', 'RIDGE', 'LASSO']):
            temp = toi[ (toi['Regression type']==reg) & (toi['Metric'] =='MSE')]
            min_row = toi.loc[temp['Value'].idxmin()]
            p = min_row['Complexity']
            lam = min_row['lambda']   
            
            p -= 5
            
            fit_best.run_fit(int(p), reg, lam)
            fit_best.evaluate_model()
            z = fit_best._design.dot(fit_best.par)             
            terrain_fit = z.reshape(zl)

            plt.subplot(2,2, 2+i)
            plt.title(reg + ' | MSE = %.2E\n p = %i; $\lambda$ = %.1E' %(fit_best.mse, p, lam))
            c = plt.imshow(terrain_fit, cmap='gist_earth', vmin=0.9*tmin, vmax=1.1*tmax)
          
        plt.tight_layout()
        plt.savefig('./yellowstone1_%i_scale/fit.pdf'%red)
    
    
if __name__ == "__main__":
    start = time.perf_counter()
    main()
    finish = time.perf_counter()
    print("Elapsed time: ", finish - start, "s")
