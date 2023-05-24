# Import modules
import numpy as np
import sympy as sm

from scipy.stats import norm
import scipy.optimize as optimize
from numpy import array
import matplotlib.pyplot as plt
import plotly.graph_objs as go

from types import SimpleNamespace

# Defining symbolic variables
p      = sm.symbols('p')   # price
c      = sm.symbols('c')   # marginal cost
b      = sm.symbols('b')   # degree of substitution between products (range = 0-1)
x      = sm.symbols('x')   # Symbolic variable representing x
x_rest = sm.symbols('X_r') # Symbolic variable representing x_rest

# Define equations and functions

eq_price    = sm.Eq(p, 1-x-b*x_rest)     # demand (inverse demand function)
func_profit = p * x - c * x              # firm profit

p_from_pricefun = sm.solve(eq_price, p)  # isolate price as a variable

objective = func_profit.subs(p,p_from_pricefun[0]) # substitute price into profit to get objective

# Lambda functions for later use
objective_lambd = sm.lambdify(args=(x, x_rest, c, b), expr = objective)

# 1st order derivative of profit wrt. x to get FOC for profit max
objective_diff_self = sm.diff(objective, x)

# Best response function (lambda function)
bestresponse = sm.lambdify(args = (x, x_rest, c, b), expr = objective_diff_self)

# 2nd order derivative of profit wrt. x and x_rest to get Jacobian Matrix
bestresponse_diff_self = sm.diff(objective_diff_self, x)
bestresponse_diff_rest = sm.diff(objective_diff_self, x_rest)

# Converting the symbolic expression for the second derivative into a callable function
jac_x_self = sm.lambdify(args=(x, x_rest), expr = bestresponse_diff_self)
jac_x_rest = sm.lambdify(args=(x, x_rest, b), expr = bestresponse_diff_rest)


# Defining the function to be optimized
def h(x, c_vec, b, N):
    y = np.zeros(N)
    for i in range(N):

        y[i] = bestresponse(x[i], sum(x) - x[i], c_vec[i], b)
    return y


# Defining the Jacobian of the function to be optimized
def hp(x, b, N):
    y = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if j == i:
                # Diagonal of the Jacobian Matrix
                y[i,j] = jac_x_self(x[i], sum(x) - x[i])
            else:
                # Off-Diagonal of the Jacobian Matrix
                y[i,j] = jac_x_rest(x[i], sum(x) - x[i], b)
    return y

# Algoritm for solving market equilibrium
def solve_model(N=50, b=1, seed=2000, c_constant=9999,display=True):

    N_init = N

    # keyword c=9999 is a placeholder used to draw from a log-normal distribution
    # if c is set at any other value marginal costs are constant at c
    #
    if c_constant!=9999:
        c_vec=np.full((N,),c_constant)
    else:
        np.random.seed(seed)
        c_vec = 0.01*np.random.lognormal(mean=0,sigma=1,size=N)
    c_vec_init = c_vec.copy()

    # Setting up the initial values for x
    index    = np.array(range(N))
    x0       = np.zeros(N)
    x_nonneg = np.zeros(N, dtype=bool)

    while not all(x_nonneg):

        # Solving the optimization problem using scipy.optimize.root() function
        result = optimize.root(lambda x0: h(x0, c_vec, b, N), x0, jac=lambda x0: hp(x0, b, N))

        x0 = result.x
        x_nonneg = (x0 >= 0).astype(bool)
        
        c_vec = c_vec[x_nonneg]
        x0    = x0[x_nonneg]
        N     = np.sum(x_nonneg)
        index = index[x_nonneg]

    profit=objective_lambd(result.x, np.sum(result.x)-result.x, c_vec, b)

    # Printing the results
    if display == True:
        print(result)
        print('\nx =', result.x[0:5], '\nh(x) =', h(x0,c_vec,b,N)[0:5], '\nsum(x) =', sum(result.x), '\nmarginal cost=',c_vec[0:5],'\nprofit=', profit[0:5],'\nb= ',b,'\nN_firms =',N)

    for i in range(N_init):
        if i in index:
            continue
        else:
            x0=np.insert(x0,i,0)
            profit=np.insert(profit,i,0)

    sol = SimpleNamespace()
    sol.c_vec_init=c_vec_init
    sol.c_vec=c_vec
    sol.x0=x0
    sol.index=index
    sol.N=N
    sol.N_init=N_init
    sol.profit=profit

    return sol



# End of py-file