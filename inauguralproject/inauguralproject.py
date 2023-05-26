import numpy as np
from scipy import optimize

import pandas as pd 
import matplotlib.pyplot as plt

from types import SimpleNamespace

class HouseholdSpecializationModelClass:

    def __init__(self):
        """ setup model """

        # a. create namespaces
        par = self.par = SimpleNamespace()
        sol = self.sol = SimpleNamespace()

        # b. preferences
        par.rho = 2.0
        par.nu = 0.001
        par.epsilon = 1.0
        par.omega = 0.5 
        
        #Sets disutility parameter for doing household labor for men. 
        par.mu = 0

        # c. household production
        par.alpha = 0.5
        par.sigma = 1

        # d. wages
        par.wM = 1.0
        par.wF = 1.0
        par.wF_vec = np.linspace(0.8,1.2,5)

        # e. targets
        par.beta0_target = 0.4
        par.beta1_target = -0.1

        # f. solution
        sol.LM_vec, sol.HM_vec, sol.LF_vec, sol.HF_vec  = np.zeros(par.wF_vec.size), np.zeros(par.wF_vec.size), np.zeros(par.wF_vec.size), np.zeros(par.wF_vec.size)

        sol.beta0 = np.nan
        sol.beta1 = np.nan

        sol.alpha = np.nan
        sol.sigma = np.nan

    def calc_utility(self,LM,HM,LF,HF):
        """ calculate utility - - now supports sigma different from 1"""

        par = self.par
        sol = self.sol

        # a. consumption of market goods
        C = par.wM*LM + par.wF*LF

        # b. home production. par.sigma==0 is never required so it is skipped.
        if par.sigma== 1:
            H = HM**(1-par.alpha)*HF**par.alpha
        else:
            H=((1-par.alpha)*HM**((par.sigma-1)/par.sigma)+par.alpha*HF**((par.sigma-1)/par.sigma))**(par.sigma/(par.sigma-1))

        # c. total consumption utility
        Q = C**par.omega*H**(1-par.omega)
        utility = np.fmax(Q,1e-8)**(1-par.rho)/(1-par.rho)

        # d. disutlity of work
        epsilon_ = 1+1/par.epsilon
        TM = LM+HM
        TF = LF+HF
        disutility = par.nu*(TM**epsilon_/epsilon_+TF**epsilon_/epsilon_)+par.mu*np.fmax(par.wM/par.wF,1)*HM
        return utility - disutility

    def solve_discrete(self,do_print=False):
        """ solve model discretely """
        
        par = self.par
        sol = self.sol
        opt = SimpleNamespace()
        
        # a. all possible choices
        x = np.linspace(0,24,49)
        LM,HM,LF,HF = np.meshgrid(x,x,x,x) # all combinations
    
        LM = LM.ravel() # vector
        HM = HM.ravel()
        LF = LF.ravel()
        HF = HF.ravel()

        # b. calculate utility
        u = self.calc_utility(LM,HM,LF,HF)
    
        # c. set to minus infinity if constraint is broken
        I = (LM+HM > 24) | (LF+HF > 24) # | is "or"
        u[I] = -np.inf
    
        # d. find maximizing argument
        j = np.argmax(u)
        
        opt.LM = LM[j]
        opt.HM = HM[j]
        opt.LF = LF[j]
        opt.HF = HF[j]
        
        # e. print
        if do_print:
            for k,v in opt.__dict__.items():
                print(f'{k} = {v:6.4f}')

        return opt

    def solve(self,do_print=False):
        """ solve model continously """
        par = self.par
        sol = self.sol 
        opt = SimpleNamespace()
        
        # a. guesses:
        #LM,HM,LF,HF
        x_guess=[4.5]*4

        # b. creating objective 
        obj = lambda x: -self.calc_utility(x[0],x[1],x[2],x[3])

        # c. creating bounds
        bounds=((1e-8,24-1e-8),(1e-8,24-1e-8),(1e-8,24-1e-8),(1e-8,24-1e-8))

        # c.alternative: Uses SLSQP instead of Nelder-Mead to minimize function
        #time_constraint = lambda x: x[0]+x[1]-24 + x[2]+x[3]-24
        #constraints = ({'type':'ineq','fun':time_constraint})
        #res = optimize.minimize(obj,x_guess,method='SLSQP',bounds=bounds,constraints=constraints)
        
        # d. creating result element and extracting values from it
        res = optimize.minimize(obj,x_guess,method='Nelder-Mead',bounds=bounds) 
        opt.LM, opt.HM, opt.LF, opt.HF = res.x

        # e. print
        if do_print:
            for k,v in opt.__dict__.items():
                print(f'{k} = {v:6.4f}')

        return opt   

    def solve_wF_vec(self,discrete=False):
        """ solve model for vector of female wages """
        par = self.par
        sol = self.sol
           
        #0.  Loops through values of wF_vec
        for i,wF in enumerate(par.wF_vec): 
            self.par.wF = wF  

            # a. guesses:
            #LM,HM,LF,HF
            LM_guess=5
            HM_guess=5
            LF_guess=5
            HF_guess=5
            x_guess=[LM_guess,HM_guess,LF_guess,HF_guess]

            # b. creating objective 
            obj = lambda x: -self.calc_utility(*x)

            # c. creating bounds
            bounds=((1e-8,24-1e-8),(1e-8,24-1e-8),(1e-8,24-1e-8),(1e-8,24-1e-8))
            
            # c.alternative: Uses SLSQP instead of Nelder-Mead to minimize function        
            #time_constraint = lambda x: x[0]+x[1]-24 + x[2]+x[3]-24
            #constraints = ({'type':'ineq','fun':time_constraint})
            #res = optimize.minimize(obj,x_guess,method='SLSQP',bounds=bounds,constraints=constraints)       
            
            # d. creating result element and extracting values from it
            res = optimize.minimize(obj,x_guess,method='Nelder-Mead',bounds=bounds) 
            sol.LM_vec[i] = res.x[0]
            sol.HM_vec[i] = res.x[1]
            sol.LF_vec[i] = res.x[2]
            sol.HF_vec[i] = res.x[3]
        return sol

    def run_regression(self):
        """ run regression """

        par = self.par
        sol = self.sol
        
        #a. Generates relevant dependent and explanatory variables
        x = np.log(par.wF_vec)
        y = np.log(sol.HF_vec/sol.HM_vec)
        A = np.vstack([np.ones(x.size),x]).T

        #b. Returns beta0 and beta_1
        sol.beta0,sol.beta1 = np.linalg.lstsq(A,y,rcond=None)[0]
        return sol
    
    def calc_deviation(self,alpha,sigma):
        """ For a given alpha and sigma, returns squared deviation from realistic parameter values"""
        par = self.par
        sol = self.sol

        #a. Sets parameters and updates alpha and sigma
        par.alpha=alpha
        par.sigma=sigma

        #b. For a given alpha and sigma simulate optimal household behavior
        self.solve_wF_vec()

        #c. For a given household behavior run regression to find beta_0 and beta_1
        self.run_regression()

        #d. For beta_0 and beta_1 calculate sq. dev. from target parameters
        test=(par.beta0_target-sol.beta0)**2+(par.beta1_target-sol.beta1)**2
        return test

    def estimate(self):
        """ Minimize sq. dev. from realistic parameter values wrt. alpha and sigma """

        par = self.par
        sol = self.sol  

        # a. use guesses:
        alpha_guess=0.5
        sigma_guess=1
        guess=[alpha_guess,sigma_guess]

        # b. creating objective
        obj= lambda x: self.calc_deviation(*x) 

        # c. creating bounds
        bounds=((1e-8,1-1e-8),(1e-8,3-1e-8))

        # d. creating result element and extracting values from it
        res = optimize.minimize(obj,guess,method='Nelder-Mead',bounds=bounds) 
        sol.alpha = res.x[0]
        sol.sigma = res.x[1]
        return sol
    
    def calc_deviation_5(self,mu,sigma):
        """ For a given mu and sigma, returns squared deviation from realistic parameter values"""
        par = self.par
        sol = self.sol

        #a. Sets parameters and updates mu and sigma
        par.mu=mu
        par.sigma=sigma

        #b. For a given mu and sigma simulate optimal household behavior
        self.solve_wF_vec()

        #c. For a given household behavior run regression to find beta_0 and beta_1
        self.run_regression()

        #d. For beta_0 and beta_1 calculate sq. dev. from target parameters
        test=(par.beta0_target-sol.beta0)**2+(par.beta1_target-sol.beta1)**2
        return test

    def estimate_5(self):
        """ Minimize sq. dev. from realistic parameter values wrt. mu and sigma """

        par = self.par
        sol = self.sol  

        # a. use guesses:
        mu_guess=0.5
        sigma_guess=1
        guess=[mu_guess,sigma_guess]

        # b. creating objective
        obj= lambda x: self.calc_deviation_5(*x) 

        # c. creating bounds
        bounds=((1e-8,1-1e-8),(1e-8,3-1e-8))

        # d. creating result element and extracting values from it
        res = optimize.minimize(obj,guess,method='Nelder-Mead',bounds=bounds) 
        sol.mu, sol.sigma = res.x[0], res.x[1]
        return sol