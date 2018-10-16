###############################################################################
# LIBRARY OF FUNCTIONS TO PLOT STATISTICS OF ENSEMBLES
# AUTHORS: Man-Yau Chan (1,2) , Biing Han Chia (1,3), Li Yang Tan (1,3)
# ORGANIZATIONS: 1. Team NUS_Singapore_Sci, iGEM 2018 
#                2. The Pennsylvania State University, Department of Meteorology
#                   and Atmospheric Sciences
#                3. National University of Singapore, Special Programme in
#                   Science
###############################################################################
# PURPOSE: Library of functions to plot ensemble statistics
###############################################################################
# Date of last update: Oct 16, 2018
###############################################################################

import numpy as np
import funclib_model as mod
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
from math import pi

# Function to plot out the ensemble concentration 2d histograms
def conc_hist2d_GC( all_conc_ens,  fig_C, all_true_conc):

    # Handling the C field
    # Ploting grid is 3 by 2 
    names = ['substrate mRNA','enzyme-substrate complex','enzyme','product mRNA','mCherry','GFP-mCherry']
    axes = [ fig_C.add_subplot(3,2,i+1) for i in range(6) ]
    fig_C.delaxes(axes[5])
    concs0 = all_conc_ens[:,5]
    bin_ranges = [ np.linspace(  0, 120, 21),\
                   np.linspace( 40, 100, 21),\
                   np.linspace(  0,  40, 21),\
                   np.linspace(  0,  80, 21),\
                   np.linspace(  0,3000, 21),\
                   np.linspace(  0,3000, 21)  ]


    for i in range(6):
        if i != 5:
            concs1 = all_conc_ens[:,i]
            concs = np.array( [concs0, concs1] )
            conc_mean = np.mean( concs, axis=1 )
            cov = np.matrix( np.cov( concs ) )
            sig0 = np.sqrt( cov[0,0])
            sig1 = np.sqrt( cov[1,1])
            inv_cov = np.matrix( np.linalg.inv( cov) )
            bin_range = [bin_ranges[5], bin_ranges[i] ]
            ax = axes[i]
            h, xedges, yedges, image = ax.hist2d( concs[0],concs[1], bins = bin_range, 
                                                  cmap=plt.cm.Greens) #, normed=True)
            cbar = plt.colorbar( image, ax=ax)
            ax.set_xlabel('[GFP-mCherry]')
            ax.set_ylabel('['+names[i]+']')
            ax.plot( [all_true_conc[5]], [all_true_conc[i]], '.r', markersize=12)
            

    return 

# Function to plot out the ensemble concentration 2d histograms
def conc_hist2d_C( all_conc_ens,  fig_C, all_true_conc):

    # Handling the C field
    # Ploting grid is 3 by 2 
    names = ['substrate mRNA','enzyme-substrate complex','enzyme','product mRNA','mCherry','GFP-mCherry']
    axes = [ fig_C.add_subplot(3,2,i+1) for i in range(6) ]
    fig_C.delaxes(axes[4])
    concs0 = all_conc_ens[:,4]
#    concs0 = concs0[ all_conc_ens[:,4] < np.percentile( all_conc_ens[:,4], 95 )]
    bin_ranges = [ np.linspace(  0, 120, 21),\
                   np.linspace( 40, 100, 21),\
                   np.linspace(  0,  40, 21),\
                   np.linspace(  0,  80, 21),\
                   np.linspace(  0,3000, 21),\
                   np.linspace(  0,3000, 21)  ]

    for i in range(6):
        if i != 4:
            concs1 = all_conc_ens[:,i]
            concs = np.array( [concs0, concs1] )
            conc_mean = np.mean( concs, axis=1 )
            cov = np.matrix( np.cov( concs ) )
            sig0 = np.sqrt( cov[0,0])
            sig1 = np.sqrt( cov[1,1])
            inv_cov = np.matrix( np.linalg.inv( cov) )
            bin_range = [bin_ranges[4], bin_ranges[i]]      
            ax = axes[i]
            h, xedges, yedges, image = ax.hist2d( concs[0],concs[1], bins = bin_range, 
                                                  cmap=plt.cm.Greens) #, normed=True)
            cbar = plt.colorbar( image, ax=ax)
            ax.set_xlabel('[mCherry]')
            ax.set_ylabel('['+names[i]+']')
            ax.plot( [all_true_conc[4]], [all_true_conc[i]], '.r', markersize=12)

    return 


# Function to plot out the ensemble concentration histograms to examine gaussianity
def conc_gaussian_check( all_conc_ens, fig ):

    # Plot grid is 3 by 2
    names = ['substrate mRNA','enzyme + substrate mRNA complex','enzyme','product mRNA','mCherry','GFP-mCherry']
    axes = [ fig.add_subplot(3,2,i+1) for i in range(6) ]
    
    for i in range(6):
        conc_ens = all_conc_ens[:,i]
        ax = axes[i]
        conc_bins = np.linspace( np.min( conc_ens), np.percentile( conc_ens, 95), 51 )
        conc_ens1 = conc_ens[ conc_ens < np.percentile( conc_ens, 95) ]
        conc_ens = conc_ens1
        histo_pdf, bin_edges, patches = ax.hist( conc_ens, bins = conc_bins, normed=True, color='lightgrey' )
        # Compute mean and std dev
        conc_mean = np.mean( conc_ens )
        conc_std = np.std( conc_ens )
        # Overlay Gaussian
        g_pdf = np.exp( -np.power(bin_edges-conc_mean,2)/(2*conc_std*conc_std) ) /np.sqrt( 2 * pi * conc_std * conc_std )
        ax.plot( bin_edges, g_pdf,'-k' )
        ax.set_title( names[i] )
#        ax.set_yscale('log')

    return 


