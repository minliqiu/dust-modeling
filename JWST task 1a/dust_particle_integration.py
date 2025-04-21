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
L_Sun = 3.828e26 # solar luminosity in [watts]
m_J = 1.898e27 # Jupiter mass in [kg]
R_J = 7.1492e7 # Jupiter radius in [m]
a_J = 7.78479e8 # Jupiter semi-major axis in [m]
m_E = 5.9722e24 # Earth mass in [kg]
R_E = 6.371e6 # Earth radius in [m]


def integration(variable):
    
    k1, k2, k_ap, beta, k_init, e_initial, inc_initial, endtime, N = variable
    
    # define variables
    m_Star = k1 * m_Sun
    R_Star = k1**0.75 * R_Sun
    R_sub = max((k1**2) * 4 * R_Sun, R_Star)
    # L_Star = k1**3.5 * L_Sun

    R_sub_To_R_Sun = R_sub/R_Sun
    
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

    CJ_factor = G*(m_Star+m_Planet)/a_Planet
    CJ_final = []
    x_p_f_xyz, v_p_f_xyz = [], [] # planet final position and velocity
    x_d_f_xyz, v_d_f_xyz = [], [] # dust final position and velocity
    final_fate = []
    lifetime = []
    a_d_f, e_d_f = [], []
    vrel_f, mu_f = np.array([np.nan]), np.array([np.nan]) # NaN by default when dust-planet collision doesn't happen

    # get relative velocity and angle at collison
    def compute_collision_info(r1, v1, r2, v2, R_target):
        r_rel = r2 - r1
        v_rel = v2 - v1

        r = np.linalg.norm(r_rel)
        v = np.linalg.norm(v_rel)

        mu = np.dot(r_rel, v_rel) / (r * v)

        # Predict closest approach assuming straight-line motion
        c1 = np.dot(r_rel, v_rel) / v**2
        c2 = (r**2 - R_target**2) / v**2
        if c1**2 - c2 >= 0:
            tc = -c1 + np.sqrt(c1**2 - c2)
            r_closest = r_rel + v_rel * tc
            mu = np.dot(r_closest, v_rel) / (np.linalg.norm(r_closest) * v)

        return v, mu

    # collision function
    def collision_function(sim_pointer, collision):
        hash_Star = str(ps['Star'].hash)
        hash_Planet = str(ps['Planet'].hash)

        simcontps = sim_pointer.contents.particles # get simulation object from pointer
        hash_p1 = str(simcontps[collision.p1].hash)
        hash_p2 = str(simcontps[collision.p2].hash)

        CJ_final.append(get_jacobi_const(sim, ps[2]))
        x_p_f_xyz.append(ps['Planet'].xyz)
        v_p_f_xyz.append(ps['Planet'].vxyz)
        x_d_f_xyz.append(ps[2].xyz)
        v_d_f_xyz.append(ps[2].vxyz)
        lifetime.append(sim.t/yr)
        ps["Star"].m = m_Star*(1-beta)
        a_d_f.append(ps[2].orbit(primary=ps['Star']).a)
        e_d_f.append(ps[2].orbit(primary=ps['Star']).e)
        ps["Star"].m = m_Star


        # p1 not dust              
        if hash_p1 == hash_Star:
            final_fate.append('sublimation')
            print ('Sublimation:', hash_p2, 'at %.5f'%(sim.t/yr), '[yr]')
            j = 2 # remove p2 (dust)

        elif hash_p1 == hash_Planet:
            final_fate.append('collision')
            print ('Hit the Planet:', hash_p2, 'at %.5f'%(sim.t/yr), '[yr]')
            
            planet_xyz = np.array(ps['Planet'].xyz)
            planet_vxyz = np.array(ps['Planet'].vxyz)
            dust_xyz = np.array(ps[2].xyz)
            dust_vxyz = np.array(ps[2].vxyz)
            vrel, mu = compute_collision_info(planet_xyz, planet_vxyz, dust_xyz, dust_vxyz, R_Planet)
            vrel_f[0] = vrel / 1e3  # km/s
            mu_f[0] = mu
            
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
                
                planet_xyz = np.array(ps['Planet'].xyz)
                planet_vxyz = np.array(ps['Planet'].vxyz)
                dust_xyz = np.array(ps[2].xyz)
                dust_vxyz = np.array(ps[2].vxyz)
                vrel, mu = compute_collision_info(planet_xyz, planet_vxyz, dust_xyz, dust_vxyz, R_Planet)
                vrel_f[0] = vrel / 1e3  # km/s
                mu_f[0] = mu
            
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
    a_initial = k_init*a_Planet                 # Semi-major axis
    pomega_initial = 2*np.pi*np.random.rand()   # Longitude of pericenter
    f_initial = 2*np.pi*np.random.rand()        # True anomaly
    Omega_initial = 2*np.pi*np.random.rand()    # Longitude of node
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
        mu = sim.G * ps['Star'].m * (1-beta)
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
        mu1 = sim.G * ps['Star'].m * (1-beta)
        mu2 = sim.G * ps['Planet'].m
        r1 = r-rstar
        r2 = r-rplanet
        PE = -mu1/np.sqrt(r1@r1) - mu2/np.sqrt(r2@r2) # test particle potential energy

        lz = np.cross(r,v)[-1] # component of the test particle's specific angular momentum aligned with planet's orbit normal

        CJ_sim = 2 * ps['Planet'].n * lz - 2 * (KE + PE) # jacobi constant
        
        CJ = -CJ_sim/CJ_factor # normalize to 3.

        return CJ
       
    # start integration
    Noutput = 100
    times = np.linspace(0, endtime, Noutput)

    
    CJ, a_d, e_d = np.zeros(Noutput), np.zeros(Noutput), np.zeros(Noutput)

    for i, time in enumerate(times):
        sim.integrate(time)

        if sim.N == 2:
            print ('No dusts left. Finish integration. :)')
            break
        
        CJ[i] = get_jacobi_const(sim, ps[2])
        a_d[i], e_d[i] = ps[2].orbit(primary=ps['Star']).a, ps[2].orbit(primary=ps['Star']).e

        # ejection judgement
        if get_E(sim, ps[2])>0 and ps[2].orbit(primary=ps['Star']).a > 10*a_Planet:
            final_fate.append('ejection')
            lifetime.append(sim.t/yr)
            CJ_final.append(get_jacobi_const(sim, ps[2])) # CJ.append(get_jacobi_const(sim, ps[2]))
            x_p_f_xyz.append(ps['Planet'].xyz)
            v_p_f_xyz.append(ps['Planet'].vxyz)
            x_d_f_xyz.append(ps[2].xyz)
            v_d_f_xyz.append(ps[2].vxyz)
            lifetime.append(sim.t/yr)
            ps["Star"].m = m_Star*(1-beta)
            a_d_f.append(ps[2].orbit(primary=ps['Star']).a)
            e_d_f.append(ps[2].orbit(primary=ps['Star']).e)
            ps["Star"].m = m_Star
            print ('Ejection:', str(ps[2].hash), 'at %.5f'%(sim.t/yr), '[yr]')
            sim.remove(2)
            break

    if len(final_fate)==0:
        final_fate.append('incomplete')
        lifetime.append(sim.t/yr)
        CJ_final.append(get_jacobi_const(sim, ps[2])) # CJ.append(get_jacobi_const(sim, ps[2]))
        x_p_f_xyz.append(ps['Planet'].xyz)
        v_p_f_xyz.append(ps['Planet'].vxyz)
        x_d_f_xyz.append(ps[2].xyz)
        v_d_f_xyz.append(ps[2].vxyz)
        lifetime.append(sim.t/yr)
        ps["Star"].m = m_Star*(1-beta)
        a_d_f.append(ps[2].orbit(primary=ps['Star']).a)
        e_d_f.append(ps[2].orbit(primary=ps['Star']).e)
        ps["Star"].m = m_Star

    
    # outcome
    para = np.array([[k1, k2, k_ap, R_sub_To_R_Sun, R_Planet/R_Sun,
                      beta, a_initial/a_Planet, e_initial, inc_initial, Omega_initial, pomega_initial, f_initial,
                      final_fate[0], lifetime[0], x_p_f_xyz[0], v_p_f_xyz[0], x_d_f_xyz[0], v_d_f_xyz[0], CJ_final[0],
                      a_d_f[0], e_d_f[0], vrel_f[0], mu_f[0], CJ, a_d, e_d ]], dtype=object)
    
    paralabels = ["m_Star/m_Sun", "m_Planet/m_J", "a_p/R_Sun", "R_sub/R_Sun", "R_Planet/R_Sun",
                  "beta", "a_d_i/a_p", "e_d_i", "inc_d_i", "Omega_d_i", "pomega_d_i", "f_d_i",  
                  "final_fate", "lifetime[yr]", "x_p_f_xyz[m]", "v_p_f_xyz[m/s]", "x_d_f_xyz[m]",
                  "v_d_f_xyz[m/s]", "CJ_final", "a_d_f", "e_d_f", "vrel_f[km/s]", "mu", "CJ", "a_d", "e_d" ]
    df_para_new = pd.DataFrame(para, columns = paralabels)
    
    return df_para_new


