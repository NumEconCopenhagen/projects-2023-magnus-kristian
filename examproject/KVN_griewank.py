
# KVN's attempt at solving the griewank function

# Define function with algoritm to solve the griewank function
def griewank_solver(set_K_init=10):
    """
    griewank_solver
        Solution algoritm for the griewank function
    Args
    ----------
    set_K_init : int
        Number of initial warm-up tries
    Returns
    ----------
    x_star_best : array
        Best solution found
    f_star_best : float
        Best value found    
    """
    #1. bounds and tolerance

    tol_tau = 1e-8
    bounds = [(-600,600),(-600,600)]

    #2. warm-up iterations and maximum iterations
    K_init = set_K_init
    K_max  = 1_000

    #3. algoritm: loop K times

    x_vec = np.empty((K_max,2))
    x_init= np.empty((K_max,2))

    x_star= np.empty((K_max,2))
    x_star_best = np.empty(2)

    f_star = np.empty((K_max))
    f_star_best = np.empty(2)

    xi_vec = np.empty(K_max)

    flag_converged = False

    for k in range(K_max):

        #3.a draw random x
        x_vec[k,:] = np.random.uniform(bounds[0][0],bounds[0][1],2)

        #3.b. skip directly to optimizer if still in warm-up phase
        if (k < K_init):

            # initial in warm-up phase
            x_init[k,:] = x_vec[k,:]

        if not (k < K_init):

            #3.c set xi
            xi_vec[k] = 0.50 * 2 / (1+np.exp(-k/K_init)/100)

            #3.d. set initial
            x_init[k,:] = xi_vec[k]*x_vec[k,:] + (1-xi_vec[k])*x_star_best

        #3.e optimize: minimize griewank with BFGS and tol of tol_tau, return x_star
        sol = optimize.minimize(griewank,x_vec[k,:],method='BFGS',tol=tol_tau)

        x_star[k,:] = sol.x
        f_star[k]   = sol.fun  

        # 3.f update x_star_best and f_star_best if better than previous
        if k == 0 or f_star[k] < f_star_best:
            x_star_best = x_star[k,:]
            f_star_best = f_star[k]

        #3.g if f_star_best < tol_tau, break
        if f_star_best < tol_tau:
            flag_converged = True
            break

    # 4 print best result and convergence flag
    print(f'Full solution \n {sol}')
    print(f'Converged: {flag_converged}')
    print(f'Best result: {f_star_best:.2f} at x = {x_star_best}')

    return flag_converged, x_star, f_star, x_init, x_vec, xi_vec


griewank_solver()

