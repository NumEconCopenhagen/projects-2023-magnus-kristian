import numpy as np
import sympy as sm
import array

from scipy.stats import norm
import scipy.optimize as optimize
from numpy import array
import matplotlib.pyplot as plt
import plotly.graph_objs as go

from types import SimpleNamespace

# Importing necessary libraries
import numpy as np
import scipy.optimize as optimize
import sympy as sm

# Defining symbolic variables for the optimization problem
x = sm.symbols('x')  # Symbolic variable representing x
x_rest = sm.symbols('x_rest')  # Symbolic variable representing x_rest
c=sm.symbols('c')

# Defining the objective function
#penalty = 100  # choose a suitable penalty value
objective = (1 - x - x_rest - c) * x

objective_lambd=sm.lambdify(args=(x,x_rest,c),expr=objective)

# Taking the first derivative of the objective function w.r.t x
obj_dif = sm.diff(objective, x)

# Converting the symbolic expression for the derivative into a callable function
best = sm.lambdify(args=(x, x_rest,c), expr=obj_dif)

# Taking the second derivative of the objective function w.r.t x
best_dif = sm.diff(obj_dif, x)

# Converting the symbolic expression for the second derivative into a callable function
jac_x = sm.lambdify(args=(x, x_rest), expr=best_dif)

# Taking the second derivative of the objective function w.r.t x_rest
best_dif = sm.diff(obj_dif, x_rest)

# Converting the symbolic expression for the second derivative into a callable function
jac_x_rest = sm.lambdify(args=(x, x_rest), expr=best_dif)

# Defining the function to be optimized
def h(x, c_vec, N):
    y = np.zeros(N)
    for i in range(N):
        # Evaluating the first derivative of the objective function at x[i]
        #c_rand = norm.rvs(loc=0, scale=0.001)
        y[i] = best(x[i], sum(x) - x[i], c_vec[i])
    return y

# Defining the Jacobian of the function to be optimized
def hp(x,N):
    y = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if j == i:
                # Evaluating the second derivative of the objective function w.r.t x[i]
                y[i,j] = jac_x(x[i], sum(x) - x[i])
            else:
                # Evaluating the second derivative of the objective function w.r.t x_rest[i]
                y[i,j] = jac_x_rest(x[i], sum(x) - x[i])
    return y

#Initial løsning
# Setting up the parameters for the optimization problem

def mynewfun(N=50, seed=2000):
    # N      = N  # Number of variables
    N_init = N

    # c_vec = np.random.normal(loc=0,scale=0.50,size=N)
    # c_vec=c_vec**8

    np.random.seed(seed)
    c_vec = 0.01*np.random.lognormal(mean=0,sigma=1,size=N)
    c_vec_init = c_vec.copy()

    # Setting up the initial values for x
    index    = np.array(range(N))
    x0       = np.zeros(N)
    x_nonneg = np.zeros(N, dtype=bool)

    while not all(x_nonneg):

        # Solving the optimization problem using scipy.optimize.root() function
        result = optimize.root(lambda x0: h(x0,c_vec,N), x0, jac=lambda x0: hp(x0,N))

        x0 = result.x
        x_nonneg = (x0 >= 0).astype(bool)
        
        c_vec = c_vec[x_nonneg]
        x0    = x0[x_nonneg]
        N     = np.sum(x_nonneg)
        index = index[x_nonneg]

    profit=objective_lambd(result.x,np.sum(result.x)-result.x,c_vec)
    # Printing the results
    print(result)
    print('\nx =', result.x[0:5], '\nh(x) =', h(x0,c_vec,N)[0:5], '\nsum(x) =', sum(result.x), '\nmarginal cost=',c_vec[0:5],'\nprofit=', profit[0:5],'\nN_firms =',N)
    
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