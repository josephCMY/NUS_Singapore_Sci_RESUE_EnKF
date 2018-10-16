###############################################################################
# LIBRARY OF FUNCTIONS FOR SIMPLE RESCUE KINEMATICS MODEL
# AUTHORS: Man-Yau Chan (1,2) , Biing Han Chia (1,3), Li Yang Tan (1,3)
# ORGANIZATIONS: 1. Team NUS_Singapore_Sci, iGEM 2018
#                2. The Pennsylvania State University, Department of Meteorology
#                   and Atmospheric Sciences
#                3. National University of Singapore, Special Programme in
#                   Science
###############################################################################
# PURPOSE: Library of functions to simulate the reactions in a mammalian cell
#          with the RESCUE system.
###############################################################################
# AVAILABLE FUNCTIONS
# 1) Function to run the model
# 2) Functions to compute more mathematically convoluted components of the model
#    See the documentation for the mathematical details of the model
###############################################################################
# DOCUMENTATION: See RESCUE_EnKF_wiki.pdf in the GitHub repository.
#                It should be available in this directory if you downloaded the
#                repository.
###############################################################################
# Date of last update: Oct 16, 2018
###############################################################################

import numpy as np;
import matplotlib as plt;


# Function to run the model
# Note that reactants is simply a list of 6 initial concentrations
# reactants list of concentrations (in order) is:
# [ substrate mRNA, enzyme-substrate mRNA complex, enzyme, product mRNA, mCherry,
#   GFP-mCherry ]
# param is a list of parameters for model, in order:
# [g_s, d_s, k_P, Keq, d_P, k_C, d_C, k_GC, d_GC]
# See documentation to determine which parameter is for which equation
# dt is the time step size
# max_time_steps is the maximum number of time steps for the integration
def run_model(reactants, param, dt, max_time_steps):
    outflag=True
    # Initialize memory
    [S, ES, E, P, C, GC, Et, St ] = [np.zeros( max_time_steps+1 ) for i in range(8)]
    # Read in initial reactants
    S[0] = reactants[0];
    ES[0] = reactants[1];
    E[0] = reactants[2];
    P[0] = reactants[3];
    C[0] = reactants[4];
    GC[0] = reactants[5];

    # Compute initial total enzyme and total substrate
    Et[0] = E[0] + ES[0]
    St[0] = S[0] + ES[0]

    # Read in parameters
    g_s = param[0];
    d_s = param[1];
    k_P = param[2];
    Keq = param[3];
    d_P = param[4];
    k_C = param[5];
    d_C = param[6];
    k_GC = param[7];
    d_GC = param[8];

    # Run the model to equilibrium
    for t in range (max_time_steps):

        # Computing temporal derivatives
        delS_t = g_s - d_s*S[t] - k_P*ES[t];
        delES = delf1(Et[t], St[t], Keq)*delS_t;
        delS = delS_t - delES;
        delE = -delES;
        delP = k_P*ES[t] - d_P*P[t];
        delC = k_C*P[t] - d_C*C[t];
        delGC = k_GC*S[t] - d_GC*GC[t];

        # Updating the concentration values
        S[t+1] = S[t] + delS*dt;
        St[t+1] = St[t] + delS_t*dt
        Et[t+1 ] = Et[t]
        ES[t+1] = ES[t] + delES*dt;
        E[t+1] = E[t] + delE*dt;
        P[t+1] = P[t] + delP*dt;
        C[t+1] = C[t] + delC*dt;
        GC[t+1] = GC[t] + delGC*dt;

        # Checking whether the system has reached equilibrium
        x = [ np.array([ S[i], St[i],  ES[i], E[i], P[i], C[i], GC[i] ] )  \
              for i in [t,t+1] ]
        diff = np.abs((x[1] - x[0])/x[0])/dt
        flags = diff > 0.01
        if( np.sum(flags) == 0):
            outflag = True
            break;
        # Stop integration if negative values detected
        if( np.sum( x[1] < 0 ) > 0):
            outflag = False
            break
        # Rerun simulation if the maximum number of time steps has been attained
        if( t == max_time_steps -1 ):
            outflag = False
            break
        # Stop integration if nan or infinities detected
        if( np.sum(np.isnan( x[1]) + np.isinf( x[1])) > 0):
            outflag = False
            break

    react_tlist = [S[:(t+1)], ES[:(t+1)], E[:(t+1)], P[:(t+1)], C[:(t+1)], GC[:(t+1)]]

    # Return only up till eqbm values
    return react_tlist, outflag

# Mathematical function needed to infer the eqbm concentrations of enzymes, substrate
# mRNA and enzyme-substrate complex
# Et and St are the total enzyme and substrate concentrations respectively
def f1(Et,St,Keq):
    return(0.5*(Et + St + Keq) - 0.5*np.sqrt((Et + St + Keq)**2 - 4*Et*St));

# Derivative of previous mathematical function wrt total substrate concentration.
# Needed for rate of change computations.
def delf1(Et,St,Keq):
    return(0.5 - (2*(Et + St + Keq) - 4*Et) / ( 4 * np.sqrt( (Et + St + Keq)**2 - 4*Et*St)) )

