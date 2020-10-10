import Poly2DFit 
import numpy as np 
import pytest
from sklearn. metrics import r2_score, mean_squared_error
from sklearn.linear_model import Ridge, LinearRegression
from additional_functions_project1 import R2, MSE
"""
run pytest command from comandline 
automatically executes all test_*.py or *_test.py files
"""

np.random.seed(2019)

def test_Poly2DFit():
    """
    testing the functionallity of Poly2DFit by try to reconstruct a 
    given parameter example for a polynomial of degree 3 i.e 10 random parameters
    for x,y in [-1, 1]
    """
    par  = 20*np.random.randn(10) + 2
    x,y = 2 * np.random.rand(2,100)  - 1 
    create_fit = Poly2DFit.Poly2DFit()
    #asign data 
    create_fit.x = x
    create_fit.y = y
    create_fit.order = 3
    create_fit.matDesign(x,y)
    design = create_fit._design
    #assigning data to fit to with X.beta
    data = design.dot(par)
    #free memory and test with new class instance
    del create_fit 
    test_fit = Poly2DFit.Poly2DFit()
    test_fit.givenData(x, y, data)

    #test OLS###########################################################################
    test_par, _ = test_fit.run_fit(3, 'OLS')
    res = np.abs(test_par -par).max()
    #test assertion to precison of 10^-9
    assert res == pytest.approx( 0 ,  abs = 1e-9 ) 
    
    #test RIDGE#########################################################################
    #only accurate for lam = 0 ?
    test_par, _ = test_fit.run_fit(3, 'RIDGE', 0)
    res = np.abs(test_par -par).max()
    #test assertion to precison of 10^-9
    assert res == pytest.approx( 0 ,  abs = 1e-9 ) 

def test_metrices():
    """
    test the MSE and R2 implementation against sklearn and numpy
    """
    x = np.random.rand(1000)
    data =2*x + 2*np.random.randn(1000) + 5
    #test MSE usage as var
    assert np.abs( MSE(data, np.mean(data)) - np.var(data) ) == pytest.approx(0)
    #test MSE usage as MSE
    assert np.abs(MSE(data, 2*x) - mean_squared_error(data, 2*x)) == pytest.approx(0)
    #test R2
    assert np.abs(R2(data, 2*x) - r2_score(data, 2*x)) == pytest.approx(0)

def test_against_sklearn():
    fit = Poly2DFit.Poly2DFit()
    fit.generateSample(1000)
    par_OLS, _  = fit.run_fit(3,'OLS')
    par_RIDGE,_  = fit.run_fit(3,'RIDGE', 0.01)
    X = fit._design
    z = fit.data
    sk_OLS = LinearRegression(fit_intercept = False).fit(X,z).coef_
    sk_RIDGE = Ridge(alpha = 0.01,fit_intercept = False, solver ='svd').fit(X,z).coef_
    assert np.abs(par_OLS -sk_OLS).max() == pytest.approx( 0 ,  abs = 1e-9 ) 
    assert np.abs(par_RIDGE -sk_RIDGE).max() == pytest.approx( 0 ,  abs = 1e-9 ) 






