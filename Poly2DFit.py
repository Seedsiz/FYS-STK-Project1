from additional_functions_project1 import MSE, R2, FrankeFunction
from plotting_functions import plot_it
import numpy as np
import subprocess
import warnings
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso

class Poly2DFit:
    """
    class which perfoms a 2D polynomial fit to given data or generated samples from the Franke function
    Class Variables:
     -dependentvariables are stored in x,y and the constructed design matrix in _design
     -data to fit in data
     -order of the polynomial to use in order
     -all perfomrance information are stored in mse, r2, variance, bias
     - parameters and theri variance are stored in par, par_var
    Class Methodes:
        -generateSample from Franke function
        -givenData input real data
        -matDesign creates a design matrix
        -_linReg and _ridgeReg calculate parameters with respectivly regression type and calculate parameter variance
        - runFit performes the fit
    """
    def __init__(self):
        """
        initialize kfold flag
        """
        self.kfold = False
        #set seed for comparability
        np.random.seed(1709)
        self.mse = 0
        self.mse_train = 0
        self.r2 = 0
        self.variance = 0
        self. bias = 0

    def givenData(self, x, y, f):
        """
        stores given 2D data in class
        x,y are dependent variables, f= f(x,y)
        """
        self.x = x
        self.y = y
        self.data = f

    def matDesign (self, x , y , indVariables = 2):
        '''This is a function to set up the design matrix
        the inputs are :dataSet, the n datapoints, x and y data in a nx2 matrix
                        order, is the order of the coefficients,
                        indVariables, the number of independant variables
        the outputs are X
        '''

        # find the number of coefficients we will end up with
        num_coeff = int((self.order + 1)*(self.order + 2)/2)

        #find the number of rows in dataSet
        n = np.shape(x)[0]
        # create an empty matrix of zeros
        self._design = np.zeros((n,num_coeff))

        #fast assignment
        temp = self._design.T
        current_col = 0

        for p in range(self.order + 1):
            for i in range(p + 1):
                temp[current_col] = x**(p-i) * y**(i)
                current_col += 1

        self._design = temp.T

    def _linReg(self):
        """
        calculates the estimated parameters of an OLS
        outputs variance as the diagonal entries of (X^TX)^-1
        """
        XTX = np.array(self._design.T.dot(self._design), dtype=np.float64)
        #try to use standard inversion, otherwise use SVD
        try:
            inverse = np.linalg.inv(XTX)
            bad

        except:
            #warnings.warn("Singular Matrix: Using SVD", Warning)
            U, S, VT = np.linalg.svd(XTX)
            sinv = np.zeros(S.shape)
            sinv[np.abs(S) > 10e-12 ] =  1./S[np.abs(S) > 10e-12 ]
            inverse = (VT.T*sinv).dot(U.T)

        self.par_var = np.diag(inverse)
        self.confidence = 2*np.sqrt(self.par_var)
        if self.kfold:
            self.par = inverse.dot(self._design.T).dot(self.datatrain)
        else:
            self.par = inverse.dot(self._design.T).dot(self.data)

    def _ridgeReg(self):
        """
        returns the estimated parameters of an Ridge Regression with
        regularization parameter lambda
        outputs variance as the diagonal entries of (X^TX- lam I)^-1
        """
        #creating identity matrix weighted with lam
        diag = self.lam * np.ones(self._design.shape[1])
        XTX_lam = self._design.T.dot(self._design) + np.diag(diag)
        #try to use standard inversion, otherwise use SVD
        try:
            inverse = np.linalg.inv(XTX_lam)
        except:
            warnings.warn("Singular Matrix: Using SVD", Warning)
            U, S, VT = np.linalg.svd(XTX_lam)
            sinv = np.where( (np.abs(S) < 10e-25) | (S == 0), 0, 1./S)
            inverse = (VT.T*sinv).dot(U.T)

        self.par_var = np.diag(inverse)
        self.confidence = 2*np.sqrt(self.par_var)
        if self.kfold:
            self.par = inverse.dot(self._design.T).dot(self.datatrain)
        else:
            self.par = inverse.dot(self._design.T).dot(self.data)

    def _lasso(self):
        """
        Creates a model using Lasso, Returns the estimated output z
        """
        lasso = Lasso(alpha=self.lam, max_iter=10e5,tol=0.001, precompute = True, fit_intercept = False)
        # creates the Lasso parameters, beta
        if self.kfold:
            clf =  lasso.fit(self._design,self.datatrain)
            self.par = clf.coef_
        else:
            clf =  lasso.fit(self._design,self.data)
            self.par = clf.coef_
        self.par_var = 0

    def run_fit(self, Pol_order, regtype, lam = 0.1):
        """
        perfomes the fit of the data to the model given as design matrix
        suportet regtypes are 'OLS', 'RIDGE'
        lam is ignored for OLS
        returns fit parameters and their variance
        """
        self.order = Pol_order
        self.lam = lam
        self.regType = regtype
        self.model = np.zeros(self.data.shape)

        if self.kfold:
            Poly2DFit.matDesign(self, self.xtrain, self.ytrain)
        else:
            Poly2DFit.matDesign(self, self.x, self.y)

        if regtype == 'OLS':
            Poly2DFit._linReg(self)
        if regtype == 'RIDGE':
            Poly2DFit._ridgeReg(self)
        if regtype == 'LASSO':
            Poly2DFit._lasso(self)

        return self.par, self.par_var

    def kfold_cross(self, Pol_order, regtype, lam = 0.1, k= 1):
        """
        runs the k-Fold cross-validation on the given data
        sets the training example afterwards to self.x, self.y, self.data
        sets kfold flag to True
        """
        self.k = k + 1
        self.kfold = True
        #make sure all is set to 0
        self.mse = 0
        self.mse_train = 0
        self.r2 = 0
        self.variance = 0
        self. bias = 0
        self.beta_mean = 0

        np.random.seed(0)
        np.random.shuffle(self.x)
        np.random.seed(0)
        np.random.shuffle(self.y)
        np.random.seed(0)
        np.random.shuffle(self.data)

        x_folds = np.array(np.array_split(self.x, k+1))
        y_folds = np.array(np.array_split(self.y, k+1))
        data_folds = np.array(np.array_split(self.data, k+1))

        for i in range(k + 1):

            self.xtrain = np.concatenate(np.delete(x_folds, i , 0))
            self.xtest  = x_folds[i]
            self.ytrain = np.concatenate(np.delete(y_folds, i , 0))
            self.ytest  = y_folds[i]
            self.datatrain = np.concatenate(np.delete(data_folds, i , 0))
            self.datatest  = data_folds[i]

            Poly2DFit.run_fit(self, Pol_order, regtype, lam)
            self.mse_train += Poly2DFit.evaluate_model(self, self.k)

    def evaluate_model(self, k = 1):

        """
        -calculates the MSE
        -calcualtes the variance and bias of the modell
        returns the modelpoints
        """
        p = self.par.shape[0]


        if self.kfold:
            #model with training input
            model_train = self._design.dot(self.par)
            MSE_train = MSE(self.datatrain, model_train)

            Poly2DFit.matDesign(self, self.xtest, self.ytest)
            #model with training and test input
            model_test = self._design.dot(self.par)

            expect_model = np.mean(model_test)
            mse_temp = MSE(self.datatest, model_test)
            self.mse += mse_temp/self.k
            self.r2 += R2(self.datatest, model_test)/self.k


            #self.bias = MSE(FrankeFunction(self.xtest, self.ytest), expect_model) # explain this in text why we use FrankeFunction
            var_temp = MSE(model_test, expect_model)
            self.variance += var_temp/self.k
            #alternative implementaton
            # MSE = bias + variance + data_var <-> bias = MSE - varinace - data_var
            
            self.bias += (mse_temp - var_temp - np.var(self.datatrain))/self.k
            
            #find the mean beta of all individual beta 
            self.beta_mean += (self.par)/self.k
        
            #returning the  weighted MSE on training data
            return MSE_train/self.k

        else:

            self.model = self._design.dot(self.par)

            expect_model = np.mean(self.model)

            self.mse = MSE(self.data, self.model)
            self.r2 = R2(self.data, self.model)


            #self.bias = MSE(FrankeFunction(self.x, self.y), expect_model) # explain this in text why we use FrankeFunction
            self.variance = MSE(self.model, expect_model)
            self.bias = self.mse - self.variance - np.var(self.data)

    def generateSample(self, n, mean = 0., var = 1):
        """
        This function creates a sample [x,y,z] where x,y are uniform random numbers [0,1)
        and z = f(x,y) + eps with f the Franke function and eps normal distributed with mean and var
        This function is used for bench marking
        """
        #use same random numbers each time to make evaulating easier
        #create x and y randomly
        self.x, self.y = np.random.rand(2,n)

        #pass the x and y data into the Franke function this will be used later in evaluating the model
        self.data = FrankeFunction(self.x, self.y) + np.sqrt(var)*np.random.randn(n) + mean

    def plot_function(self):
        """
        This functions:
        -plots the x,y and franke function data in a scatter plot
        -plots the x,y and model in a triangulation plot
        """
        self.plot_function = plot_it(self.x, self.y, self.model, self.data)


    def store_information(self, filepath, filename):
        """
        stores information about parameters and score functions
        """
        try:
            f = open(filepath + "/" + filename  + ".txt",'w+')
        except:
            subprocess.call(["mkdir", "-p", filepath ])
            f = open(filepath + "/"+ filename + ".txt",'w+')

        f.write("    Perfomance of %s regression with  %i parameters \n:" %(self.regType, len(self.par)))

        if self.regType != 'OLS':
            f.write("Regularization parameter lambda = %f\n" %self.lam)

        if self.kfold:
            f.write("k-fold cross-validation with %i runs \n" %self.k)

        f.write("MSE = %.4f \t R2 = %.4f \t Bias(model)=%.4f \t Variance(model) =%.4f \n" %(self.mse, self.r2, self.bias, self.variance))
        f.write("Parameter Information:\n")
        for i in range(len(self.par)):
            f.write("beta_%i = %.4f +- %.4f\n" %(i, self.par[i], np.sqrt(self.par_var[i])) )
        f.close()
