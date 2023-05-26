
# Python file for solving the exam project
# Introduction to programming and numerical analysis 2023

# import packages
import numpy as np
import sympy as sm
import math

from scipy import linalg
from scipy import optimize  

import matplotlib.pyplot as plt

from IPython.display import display

from types import SimpleNamespace  

# Classes and Functions

# General 0

# [None]

# Problem 1

class OptimalTaxationModelClass:

    def __init__(self):

        """ setup model """
        # a. create namespaces
        par  = self.par  = SimpleNamespace()
        sol  = self.sol  = SimpleNamespace()
        flag = self.flag = SimpleNamespace()

        # b. set parameters
        par.sigma   = 1.001       # elasticity of substitution
        par.rho     = 1.001       # ??? of consumption
        par.alpha   = 0.5         # share parameter
        par.v       = 1/(2*16**2) # ??? disutility of labor
        par.epsilon = 1.0         # ??? of labor
        par.kappa   = 1.0         # consumer cash on hand
        par.wage    = 1.0         # real wage rate
        par.eps     = 1e-8        # tolerance level (lower bound)

        # c. choice variables
        sol.tau     = np.NaN      # tax rate             range = [eps,1-eps]
        sol.G       = np.NaN      # government spending  range = [eps*wage*eps,(1-eps)*wage*24]
        sol.L       = np.NaN      # labor supply         range = [eps,24]
        sol.C       = np.NaN      # consumption          range = [eps,kappa+(1-tau)*wage*24]

        # e. flags
        flag.found_sol_tau = False
        flag.found_sol_G   = False
        flag.found_sol_L   = False
        flag.found_sol_C   = False

    def utility(self,G,L,C):
        """
        Utility function
            for given parameters
            for given value of govenment spending (G)
            and given choice of labor supply (L) and consumption (C)
        Args:
            self (class): class containing parameters
            G (float): government spending
            L (float): labor supply
            C (float): consumption
        Returns:
            u (float): utility
        """
        
        # a. unpack
        par = self.par

        sigma   = par.sigma
        rho     = par.rho
        alpha   = par.alpha
        v       = par.v
        epsilon = par.epsilon

        # b. utility of consumption
        # b1. inner utility (depends on sigma)
        if sigma == 1:
            utility_consumption_inner = C ** (alpha) * G ** (1-alpha)
        else:
            utility_consumption_inner = ((alpha) * C ** ((sigma-1)/sigma) + (1-alpha) * G ** ((sigma-1)/sigma)) ** (sigma/(sigma-1))
        
        # b2. outer utility (depends on rho)
        if rho == 1:
            utility_consumption = np.log(utility_consumption_inner)
        else:
            utility_consumption = (utility_consumption_inner ** (1-rho) - 1) / (1-rho)
        
        # c. (dis)utility of work
        utility_work = - v * (L ** (1+epsilon) / (1+epsilon))

        # d. total utility
        u = utility_consumption + utility_work
        
        return u

    # Consumer budget constraint
    def solve_C(self,tau,L):
        """
        Consumer budget constraint
            for given values of tau and L
        Args:
            self (class): class containing parameters
            tau (float): tax rate
            L (float): labor supply
        Returns:
            C (float): consumption
        """

        # a. unpack
        par = self.par

        kappa = par.kappa
        wage = par.wage
        
        # b. consumption
        C = kappa + (1-tau) * wage * L

        # c. update flags
        flag = self.flag

        flag.found_sol_C = True

        return C

    # maximize utility wrt. laobr
    def solve_L(self,tau,G):
        """
        Optimal labor supply
            for given parameters
            for given value of tax rate (tau) govenment spending (G)
        Args:
            self (class): class containing parameters
            tau (float): tax rate
            G (float): government spending
        Returns:
            L (float): labor supply
        """

        # a. unpack
        par = self.par

        eps = par.eps

        # b. objective function
        C = lambda L: self.solve_C(tau,L)

        u = lambda L: self.utility(G,L,C(L))

        obj = lambda L: - u(L)

        # c. solve
        solution = optimize.minimize_scalar(obj,method='bounded',bounds=(eps,24))

        # d. optimal labor supply
        sol = self.sol

        sol.L = solution.x

        # e. update flags
        flag = self.flag

        flag.found_sol_L = solution.success

        return sol.L

    # balanced govenment budget
    def solve_G(self,tau):
        """
        Balanced budget government spending
            for given parameters
            for given value of tax rate (tau)
        Args:
            self (class): class containing parameters
            tau (float): tax rate
        Returns:
            G (float): government spending
        """

        # a. unpack
        par = self.par

        wage = par.wage
        eps = par.eps

        # b. objective function
        L = lambda G: self.solve_L(tau,G)

        obj = lambda G: G - tau*wage*L(G)

        # c. solve
        solution = optimize.root_scalar(obj,method='bisect',bracket=[eps*wage*eps,(1-eps)*wage*24])

        # d. balanced government spending
        sol = self.sol
        sol.G = solution.root

        # e. update flags
        flag = self.flag

        flag.found_sol_G = solution.converged

        return sol.G
    
    # Social planner problem
    def solve_tau(self):
        """
        Optimal tax rate
            for given parameters
        Args:
            self (class): class containing parameters
            tau (float): tax rate
        Returns:
            tau (float): tax rate
        """

        # a. unpack
        par = self.par

        eps = par.eps

        # b. objective function

        G = lambda tau: self.solve_G(tau)

        L = lambda tau: self.solve_L(tau,G(tau))

        C = lambda tau: self.solve_C(tau,L(tau))

        u = lambda tau: self.utility(G(tau),L(tau),C(tau))
        
        obj = lambda tau: - u(tau)

        # c. solve
        solution = optimize.minimize_scalar(obj,method='bounded',bounds=(eps,1-eps))

        # d. optimal tax rate
        sol = self.sol

        sol.tau = solution.x

        # e. update flags
        flag = self.flag

        flag.found_sol_tau = solution.success

        return sol.tau
    
# Problem 2

# [None]

# Problem 3

# [None]