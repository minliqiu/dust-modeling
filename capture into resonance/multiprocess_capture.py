import os
import numpy as np
import pandas as pd
from time import process_time
from itertools import product
from multiprocess import Pool
import threading
from capture_integration import integration
import rebound
import reboundx
import matplotlib.pyplot as plt


Degree_To_Rad = np.pi/180.


k1_array = [1.] # np.linspace(0.9, 1.1, 2)
k2_array = np.logspace(-2., 2., 20) # m_planet/m_Jupiter
k_ap_array = [10.] # standard: 10
beta_array = np.logspace(-4., -0.1, 20)
k_init = np.array([2.5])
e_initial = np.array([0.])
inc_initial = np.array([0.*Degree_To_Rad])
endtime = np.array([500000 * (365*24*3600)]) # yr = 365*24*3600 s

N_ = 1 # number of particles
N_array = np.full(N_, 1) 

input_array = []
r = product(k1_array, k2_array, k_ap_array, beta_array, k_init, e_initial, inc_initial, endtime, N_array)
for ri in r:
    input_array.append(ri)


from tabulate import tabulate

text = """
All subprocesses done.
"""

output = tabulate([[text]], tablefmt='grid')


# create an empty dataframe
paralabels = ["m_Star/m_Sun", "m_Planet/m_J", "a_p/R_Sun", "R_sub/R_Sun", "R_Planet/R_Sun", "beta", "a_d_i/a_p", "final_fate"]
df_para = pd.DataFrame(columns=paralabels)


if __name__ == "__main__":
                
    pool = Pool(processes=128) # pool number = cpu number by default

    # pool_outputs = pool.map(integration, input_array)
    for pool_output in pool.imap(integration, input_array):
        # merge data
        df_para = pd.concat([df_para, pool_output], axis=0)
        df_para = df_para.reset_index(drop = True)
        # print ('index of data:', len(df_para))
        
        # output_path='~/dust/data/result0.csv'
        # pool_output.to_csv(output_path, mode='a', header=not os.path.exists(output_path)) #, index=False)
        
    # save data
    # df_para = df_para.sort_values(by=['m_Star/m_Sun', 'm_Planet/m_J', 'a_p/R_Sun'], ascending = False)
    df_para.to_csv('~/dust/capture_into_resonance/capture_500000yr.csv', header=True, index=False)
    
    pool.close() # close() doesn't kill any process; it just closes a pipe which informs that there will be no more data coming through it.
    pool.join() # Killed processes send a signal informing their parents that they are quite dead.
    
    print (output)

# start = process_time()    
# end = process_time()
# print ('\nRunning time: %s Seconds'%(end-start))

    
