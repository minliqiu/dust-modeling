#!/usr/bin/env python
# coding: utf-8


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import rebound
import reboundx


Degree_To_Rad = np.pi/180.
AU_To_Meter = 1.496e11
yr = 365*24*3600 # s
G = 6.6743e-11 # SI units
c_light = 3e8


m_Sun = 1.99e30 # solar mass in [kg]
R_Sun = 6.957e8 # solar radius in [m]
m_J = 1.898e27 # Jupiter mass in [kg]
R_J = 7.1492e7 # Jupiter radius in [m]
a_J = 7.78479e8 # Jupiter semi-major axis in [m]
m_E = 5.9722e24 # Earth mass in [kg]
R_E = 6.371e6 # Earth radius in [m]

def integration(variable):
    
    k1, k2, k_ap, beta, k_init, e_initial, inc_initial, endtime, N = variable
    
    # define variables
    m_Star = k1 * m_Sun
    R_Star = k1 * R_Sun
    R_sub = (k1**2) * 4 * R_Sun

    m_Planet = k2 * m_J

    if k2 <= 0.41:
        R_Planet = R_E * (m_Planet/m_E)**(0.59) # low-mass planet; scale with Earth
    else:
        R_Planet = R_J * k2**(-0.04) # massive planet; scale with Jupiter

    a_Planet = k_ap * R_Sun
    
    # start rebound
    sim = rebound.Simulation()
    sim.integrator = "ias15"
    sim.G = 6.674e-11 # SI units
    sim.dt = 1.e2 # Initial timestep in sec.
    sim.N_active = 2 # Make it so dust particles don't interact with one another gravitationally

    sim.add(m = m_Star, r = R_sub, hash = "Star")
    sim.add(m = m_Planet, a = a_Planet, r = R_Planet, hash = "Planet")
    sim.move_to_com()
    ps = sim.particles

    # collision function
    CJ = []
    v_rel = [] # relative velocity between dust & planet at final timestep
    final_fate = []
    lifetime = []
    a_d_f, e_d_f = [], []

    def collision_function(sim_pointer, collision):
        hash_Star = str(ps['Star'].hash)
        hash_Planet = str(ps['Planet'].hash)

        simcontps = sim_pointer.contents.particles # get simulation object from pointer
        hash_p1 = str(simcontps[collision.p1].hash)
        hash_p2 = str(simcontps[collision.p2].hash)

        # p1 not dust              
        if hash_p1 == hash_Star:
            final_fate.append('sublimation')
            print ('Sublimation:', hash_p2, 'at %.5f'%(sim.t/yr), '[yr]')
            j = 2 # remove p2 (dust)

        elif hash_p1 == hash_Planet:
            final_fate.append('collision')
            print ('Hit the Planet:', hash_p2, 'at %.5f'%(sim.t/yr), '[yr]')
            j = 2 # remove p2 (dust)

        # p1 is dust  
        else:
            if hash_p2 == hash_Star:
                final_fate.append('sublimation')
                print ('Sublimation:', hash_p1, 'at %.5f'%(sim.t/yr), '[yr]')
                j = 1 # remove p1 (dust)
            else:
                final_fate.append('collision')
                print ('Hit the Planet:', hash_p1, 'at %.5f'%(sim.t/yr), '[yr]')
                j = 1 # remove p1 (dust)

        return j                            
    
    sim.collision = "direct"
    sim.collision_resolve = collision_function

    # radiation force & PR-drag
    rebx = reboundx.Extras(sim)
    rf = rebx.load_force("radiation_forces")
    rebx.add_force(rf)
    rf.params["c"] = 3.e8
    ps["Star"].params["radiation_source"] = 1 # set 'Star' to be the source of radiation

    # add dust particle
    np.random.seed()
    a_initial = k_init*a_Planet
#     a = amin + awidth*np.random.rand()          # Semimajor axis
    pomega_initial = 2*np.pi*np.random.rand()   # Longitude of pericenter
    f_initial = 2*np.pi*np.random.rand()        # True anomaly
    Omega_initial = 2*np.pi*np.random.rand()    # Longitude of node
#     inc = incmax*np.random.rand()               # Inclination
    sim.add(a=a_initial, e=e_initial, inc=inc_initial, Omega=Omega_initial, pomega=pomega_initial, f=f_initial) # fake add to obtain Cartesian coords
    i = 2 # ps[2] is the dust particle
    xi, yi, zi = ps[i].x, ps[i].y, ps[i].z
    vxi, vyi, vzi = np.sqrt(1-beta)*ps[i].vx, np.sqrt(1-beta)*ps[i].vy, np.sqrt(1-beta)*ps[i].vz # modify v in order for a circular orbit
    sim.remove(i)
    sim.add(x = xi, y = yi, z = zi, vx = vxi, vy = vyi, vz = vzi, hash=i) # real add
    ps[i].params["beta"] = beta

    
    # pick out dusts that have been ejected out

    # PE + KE wrt Star
    def get_E(sim, ps_i):
        rstar = np.array(ps['Star'].xyz)
        r = np.array(ps_i.xyz)
        v = np.array(ps_i.vxyz)

        KE = 0.5 * v@v # test particle kinetic energy
        mu = sim.G * ps['Star'].m
        r_ds = r - rstar
        PE = -mu/np.sqrt(r_ds@r_ds) # test particle potential energy

        E = KE + PE

        return E   
    
    def get_jacobi_const(sim, ps_i):
        rstar = np.array(ps['Star'].xyz)
        rplanet = np.array(ps['Planet'].xyz)
        r = np.array(ps_i.xyz)
        v = np.array(ps_i.vxyz)

        KE = 0.5 * v@v # test particle kinetic energy
        mu1 = sim.G * ps['Star'].m
        mu2 = sim.G * ps['Planet'].m
        r1 = r-rstar
        r2 = r-rplanet
        PE = -mu1/np.sqrt(r1@r1) - mu2/np.sqrt(r2@r2) # test particle potential energy

        lz = np.cross(r,v)[-1] # component of the test particle's specific angular momentum aligned with planet's orbit normal

        CJ = 2 * ps['Planet'].n * lz - 2 * (KE + PE) # jacobi constant

        return CJ

    # start integration
    Noutput = 1000
    times = np.linspace(0, endtime, Noutput)
    e_d, a_d, n_d, kappa, resonant_angle_d, CJ1 = np.zeros(Noutput), np.zeros(Noutput), np.zeros(Noutput), np.zeros(Noutput), np.zeros(Noutput), np.zeros(Noutput)
    
    for i, time in enumerate(times):
        sim.integrate(time)

        if sim.N == 2:
            print ('No dusts left. Finish integration. :)')
            break

        ps["Star"].m = m_Star*(1-beta)
        
        mu1 = sim.G * ps['Star'].m
        mu2 = sim.G * ps['Planet'].m
        
        e_d[i], a_d[i], n_d[i] = ps[2].e, ps[2].a, ps[2].n
        resonant_angle_d[i] = 2*ps[2].l -  ps['Planet'].l - ps[2].pomega
        # kappa[i] = ps[2].a/a_res * (2*np.sqrt(1-ps[2].e**2)-1)**2 -1
        CJ1[i] = get_jacobi_const(sim, ps[2])

        ps["Star"].m = m_Star

        # ejection judgement
        if get_E(sim, ps[2])>0:
            final_fate.append('ejection')
            print ('Ejection:', str(ps[2].hash), 'at %.5f'%(sim.t/yr), '[yr]')
            sim.remove(2)
            break

    if len(final_fate)==0:
        final_fate.append('incomplete')

    
    # outcome
    para = np.array([[ k1, k2, k_ap, (k1**2)*4, R_Planet/R_Sun, beta, a_initial/a_Planet, final_fate[0] ]])
    
    paralabels = ["m_Star/m_Sun", "m_Planet/m_J", "a_p/R_Sun", "R_sub/R_Sun", "R_Planet/R_Sun", "beta", "a_d_i/a_p", "final_fate"]
    df_para_new = pd.DataFrame(para, columns = paralabels)
    df_para_new['e_d'] = [e_d]
    df_para_new['a_d'] = [a_d]
    df_para_new['n_d'] = [n_d]
    df_para_new['phi_d'] = [resonant_angle_d]
    df_para_new['CJ1'] = [CJ1]

    return df_para_new


