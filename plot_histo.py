import funclib_enkf as enkf
import funclib_stats as stats
import numpy as np
import math as m

# PLot
super_ens_t1 = [np.loadtxt( "ens_mem_t1_%d.txt" % i ) for i in range(4) ]
super_ens_t1 = np.concatenate( super_ens_t1, axis=0)
#super_ens_t1 = np.loadtxt( "ens_mem_t1_1.txt" )
#print np.std(super_ens_t1, axis=0)

stats.hist2d( super_ens_t1, "t1" )
