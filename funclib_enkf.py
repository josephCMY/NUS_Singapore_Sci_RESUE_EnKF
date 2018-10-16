###############################################################################
# LIBRARY OF ENSEMBLE KALMAN FILTER UPDATE FUNCTIONS
# AUTHORS: Man-Yau Chan (1,2) , Biing Han Chia (1,3), Li Yang Tan (1,3)
# ORGANIZATIONS: 1. Team NUS_Singapore_Sci, iGEM 2018
#                2. The Pennsylvania State University, Department of Meteorology
#                   and Atmospheric Sciences
#                3. National University of Singapore, Special Programme in
#                   Science
###############################################################################
# PURPOSE: Library of functions to run EnKF on the RESCUE model
###############################################################################
# AVAILABLE FUNCTIONS
# 1) EnKF function based off Evensen (1994) algorithm
# 2) Function to run an ensemble of models to equilibrium. Note that the model
#    must be specified in a funclib_model.py file and that it must have the 
#    appropriate function name.
# IMPORTANT NOTE: Observations are assumed to be of GFP-mCherry and mCherry 
#                 concentrations from the model, with the same indices as those
#                 specified in the super_ens1 of the second function.
#                 I.e., mCherry concentration is index 13 and GFP-mCherry
#                 concentration is index 14
###############################################################################
# DOCUMENTATION: See RESCUE_EnKF_wiki.pdf in the GitHub repository.
#                It should be available in this directory if you downloaded the
#                repository.
###############################################################################
# REFERENCES:
# 1) Evensen (1994): Sequential data assimilation with a nonlinear
#       quasi-geostrophic model using Monte Carlo methods to forecast error
#       statistics
# 2) Evensen (2003): The Ensemble Kalman Filter: theoretical formulation and
#       practical implementation
###############################################################################
# Date of last update: Oct 16, 2018
###############################################################################

import numpy as np
import funclib_model as mod


###############################################################################
# Ensemble Kalman Filter update function
###############################################################################
# INPUTS
# eqbm_ens: 2D array of ensemble members that have been run to equilibrium.
#           dimensions -- (ens member index, params and reactants)
# obs is a 1D array of measurements.
# obs_sigma is 1D array of measurement error standard deviations
def enkf_update( eqbm_ens, obs, obs_sigma ):

    # PART 1: Kalman gain matrix computation

    # Full prior covariance matrix
    Pf = np.matrix( np.cov( np.transpose(eqbm_ens), ddof = 1) )

    # Observation operator matrix for mCherry and GFP-mCherry
    H = np.matrix((np.identity( 15 ))[13:,:])

    # Obs covariance matrix
    R = np.matrix(np.identity(2))
    R[0,0] = obs_sigma[0] * obs_sigma[0]
    R[1,1] = obs_sigma[1] * obs_sigma[1]

    # Full Kalman gain matrix
    K = np.matrix( Pf * H.T * np.linalg.inv( H*Pf*(H.T) + R ) )


    # PART 2: Generate ensemble perturbations
    prior_mean = np.mean( eqbm_ens, axis = 0)
    pert_prior_ens = eqbm_ens - prior_mean
    pert_prior_obs = pert_prior_ens[:,13:]

    # PART 3: Generate mean analysis increment
    mean_incre = np.array(K * (np.matrix( obs - prior_mean[13:])).T)[:,0]


    # PART 4: ensemble analysis increments
    n_ens = (eqbm_ens.shape)[0]
    ens_incre = eqbm_ens*0.0
    # Perform update
    for ee in range(n_ens):
        pert_obs = np.random.normal( loc = 0, scale = obs_sigma )
        incre = K * (np.matrix( pert_obs - pert_prior_obs[ee,:] )).T
        ens_incre[ee] = np.array(incre)[:,0]



    # PART 5: Generate posterior ensemble
    post_mean = np.array(prior_mean + mean_incre)
    post_pert = np.array(pert_prior_ens) + np.array(ens_incre)
    post_ens = post_mean + post_pert


    # Return posterior ensemble
    return post_ens


# Forward time integration of an ensemble
def ensemble_eqbm_runs( param_guess, param_sigma, reactant_t0_guess, reactant_t0_sigma, ens_size, dt, max_time_steps):

    # Just looking at the lengths to ease the generation of arrays to represent ensemble mems
    n_param = len(param_guess)
    n_react = len(reactant_t0_guess)

    # Draw ensemble of parameters and initial reactants from a normal distribution
    # and run the simulation
    param_ens = np.zeros( [ens_size, n_param])
    react_t0_ens = np.zeros( [ens_size, n_react])
    react_t1_ens = np.zeros( [ens_size, n_react])
    for m in range(ens_size):
        eflag=False
        while np.invert(eflag):
            # Draw for parameter
            param = np.random.normal( loc = param_guess, scale = param_sigma)
            # Draw for reactant, while ensuring reactant is never negative.
            react_t0 = np.zeros(n_react)
            for rr in range( n_react):
                if( reactant_t0_sigma[rr] != 0 ):
                    flag = True
                    while flag:
                        pert = np.random.normal( loc = reactant_t0_guess[rr],
                                                 scale = reactant_t0_guess[rr] )
                        if( pert > 0 ):
                            flag = False
                        else:
                            flag = True
                    react_t0[rr] = pert
                else:
                    react_t0[rr] = reactant_t0_guess[rr]

            # Now run the simulation for various parameters until an appropriate simulation is obtained
            react_tlist, eflag = mod.run_model( react_t0, param,  dt, max_time_steps )
#        print( "Ensemble member #%06d completed in %05d steps"% (m+1, len(react_tlist[0])))
        tmp = np.zeros(n_react)
        param_ens[m,:] = param
        for i in range( n_react ):
            react_t1_ens[m,i] = react_tlist[i][-1]
            react_t0_ens[m,i] = react_tlist[i][0]
            tmp[i] = react_tlist[i][-1]
#            [S[:(t+1)], ES[:(t+1)], E[:(t+1)], P[:(t+1)], C[:(t+1)], GC[:(t+1)]]
        print ("ens: %5i, eqbm time: %3i hrs, ES/ET: %3.1e, S/P: %3.1e, GC: %3.1e, C: %3.1e, GC/C: %3.1e"
                % ( m, int(len(react_tlist[0])*dt), tmp[1]/(tmp[1]+tmp[2]), tmp[0]/tmp[3],
                    tmp[5], tmp[4], tmp[5]/tmp[4] ) )

    # Plot 2D histograms with respect to values of GFP-Cherry and Cherry
    super_ens1 = np.zeros( [ens_size, n_param + n_react ] )
    super_ens1[:,:n_param] = param_ens
    super_ens1[:,n_param:] = react_t1_ens

    #super_ens0 = super_ens1 *0.0
    #super_ens0[:,:n_param] = param_ens
    #super_ens0[:,n_param:] = react_t0_ens


    return super_ens1
