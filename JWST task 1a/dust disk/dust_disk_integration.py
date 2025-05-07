#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
from ctypes import c_uint
import pandas as pd
import numpy as np
import rebound
import reboundx

Degree_To_Rad = np.pi/180.
AU_To_Meter = 1.496e11
yr = np.pi*1e7 # 365*24*3600 # yr in [s]
G = 6.6743e-11 # SI units
c_light = 3.e8

m_Sun = 1.99e30 # solar mass in [kg]
R_Sun = 6.957e8 # solar radius in [m]
L_Sun = 3.828e26 # solar luminosity in [watts]
T_Sun = 5772 # solar temperature in [K]
m_J = 1.898e27 # Jupiter mass in [kg]
R_J = 7.1492e7 # Jupiter radius in [m]
a_J = 7.78479e8 # Jupiter semi-major axis in [m]
m_E = 5.9722e24 # Earth mass in [kg]
R_E = 6.371e6 # Earth radius in [m]


def integration(variable):
    
    k1, k2, k_ap, beta, k_init, e_p, e_initial, inc_initial, Omega_initial, pomega_initial, endtime, N_dust = variable
        
    # define variables
    m_Star = k1 * m_Sun
    R_Star = k1**0.75 * R_Sun # Hansen & Kawaler - Stellar interiors, page 28
    T_Star = np.sqrt(m_Star/m_Sun) * T_Sun # Hansen & Kawaler - Stellar interiors, page 28
    T_melt = 1687 # silicon melting temperature in [K] # CRC Handbook of Chemistry and Physics (standard reference for material properties)
    R_melt = (T_Star/T_melt)**2 * R_Star/2
    R_sub = max(R_melt, R_Star)
    # L_Star = k1**3.5 * L_Sun
    R_sub_To_R_Sun = R_sub/R_Sun
    a_Planet = k_ap * R_Sun

    # planet M-R relation model
    # Chen & Kipping (2017) https://ui.adsabs.harvard.edu/abs/2017ApJ...834...17C/abstract
    m_Planet = k2 * m_J
    m_Planet_E = m_Planet / m_E  # mass in Earth masses
    # mass thresholds in Jupiter mass units
    k2_terr = 2.04 / (m_J / m_E)  # ~0.00642
    k2_nep = 132.7 / (m_J / m_E)  # ~0.4178

    if k2 <= k2_terr:
        R_Planet = R_E * m_Planet_E**0.28 # rocky planet; steeper
    elif k2 <= k2_nep:
        # prefactor for continuity
        factor = (2.04)**(0.28 - 0.59)
        R_Planet = R_E * factor * m_Planet_E**0.59 # ice giant / Neptune-like; radius shrinks faster with decreasing mass
    else:
        # prefactor for continuity
        factor = (2.04)**(0.28 - 0.59) * (132.7)**(0.59 - (-0.04))
        R_Planet = R_E * factor * m_Planet_E**(-0.04) # gas giant; nearly flat
    
    
    # start rebound
    sim = rebound.Simulation()
    sim.integrator = "ias15"
    sim.G = G # SI units
    sim.dt = 1.e2 # Initial timestep in sec.
    sim.N_active = 2 # Make it so dust particles don't interact with one another gravitationally

    sim.add(m = m_Star, r = R_sub, hash = "Star")
    sim.add(m = m_Planet, a = a_Planet, e = e_p, r = R_Planet, hash = "Planet")
    sim.move_to_com()
    ps = sim.particles
        
    # radiation force & PR-drag
    rebx = reboundx.Extras(sim)
    rf = rebx.load_force("radiation_forces")
    rebx.add_force(rf)
    rf.params["c"] = c_light
    ps["Star"].params["radiation_source"] = 1 # set 'Star' to be the source of radiation

    # add dust particle
    hash_array = np.zeros(N_dust)
    a_initial_array, e_initial_array, inc_initial_array, Omega_initial_array, pomega_initial_array, M_initial_array = np.zeros(N_dust), np.zeros(N_dust), np.zeros(N_dust), np.zeros(N_dust), np.zeros(N_dust), np.zeros(N_dust) 
    for i in range(2, N_dust+2):
        np.random.seed()
        a_initial = k_init*a_Planet                   # Semi-major axis
        M_initial = (i-2)/(N_dust-1) * 2*np.pi        # Mean anomolay
        if i == 2: M_initial += 1e-3 # to avoid rebound misjudges M_initial=0 particle as collision
        sim.add(a=a_initial, e=e_initial, inc=inc_initial, Omega=Omega_initial, pomega=pomega_initial, M=M_initial, primary=ps['Star']) # fake add to obtain Cartesian coords
        xi, yi, zi = ps[i].x, ps[i].y, ps[i].z
        vxi, vyi, vzi = np.sqrt(1-beta)*ps[i].vx, np.sqrt(1-beta)*ps[i].vy, np.sqrt(1-beta)*ps[i].vz # modify v in order for a circular orbit
        sim.remove(i)
        sim.add(x = xi, y = yi, z = zi, vx = vxi, vy = vyi, vz = vzi, hash=i) # real add
        ps[i].params["beta"] = beta
        
        hash_array[i-2] = ps[i].hash.value # note: ps[i].hash remains the same even when particles with smaller indeces are removed; however, ps[i] changes as a continuous array
        a_initial_array[i-2] = a_initial
        e_initial_array[i-2] = e_initial
        inc_initial_array[i-2] = inc_initial
        Omega_initial_array[i-2] = Omega_initial
        pomega_initial_array[i-2] = pomega_initial
        M_initial_array[i-2] = M_initial

    # Finds the particle in `sim.particles` with the given integer hash value.
    def find_particle_by_hash(sim, target_hash_value):
        for p in sim.particles:
            if p.hash.value == target_hash_value:
                return p
        return None # if not found, then None

    
    # Track fates
    CJ_final = np.zeros(N_dust)    
    x_p_f_xyz, v_p_f_xyz = np.zeros((N_dust, 3)), np.zeros((N_dust, 3)) # planet final position and velocity
    x_d_f_xyz, v_d_f_xyz = np.zeros((N_dust, 3)), np.zeros((N_dust, 3)) # dust final position and velocity
    final_fate = np.empty(N_dust, dtype=object)
    lifetime = np.zeros(N_dust)
    a_d_f, e_d_f = np.zeros(N_dust), np.zeros(N_dust)
    vrel_f, mu_f = np.zeros(N_dust), np.zeros(N_dust) # np.array([np.nan]), np.array([np.nan]) # NaN by default when dust-planet collision doesn't happen

    
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
        
        CJ_factor = G*(m_Star+m_Planet)/a_Planet
        CJ = CJ_sim/CJ_factor # normalize to 3.

        return CJ
    
    # get relative velocity and angle at collison
    def compute_collision_info(r1, v1, r2, v2, R_target):
        r_rel = r2 - r1
        v_rel = v2 - v1
        r = np.linalg.norm(r_rel)
        v = np.linalg.norm(v_rel)
        # Predict closest approach assuming straight-line motion
        c1 = np.dot(r_rel, v_rel) / v**2
        c2 = (r**2 - R_target**2) / v**2
        if c1**2 - c2 >= 0:
            tc = -c1 + np.sqrt(c1**2 - c2)
            r_closest = r_rel + v_rel * tc
            mu = np.dot(r_closest, v_rel) / (np.linalg.norm(r_closest) * v)
            return v, mu
        return np.nan, np.nan
    
    # collision function
    def collision_function(sim_pointer, collision):
        hash_Star = ps['Star'].hash.value
        hash_Planet = ps['Planet'].hash.value

        simcontps = sim_pointer.contents.particles # get simulation object from pointer
        hash_p1 = simcontps[collision.p1].hash.value
        hash_p2 = simcontps[collision.p2].hash.value
                
        # determine which particle is dust and its fate
        if hash_p1 == hash_Star or hash_p2 == hash_Star:
            fate = 'sublimation'
            dust_hash = hash_p2 if hash_p1 == hash_Star else hash_p1
            ps_dust_col = find_particle_by_hash(sim, dust_hash)
            ind_col = int(dust_hash-2) # find the index in hash_array
            # don't need int(np.where(hash_array==dust_hash)[0])
            j = 2 if hash_p1 == hash_Star else 1 # remove dust
            print(f'Sublimation: dust {dust_hash} at {sim.t/yr:.5f} [yr]')
        else:
            fate = 'collision'
            dust_hash = hash_p2 if hash_p1 == hash_Planet else hash_p1
            ps_dust_col = find_particle_by_hash(sim, dust_hash)
            ind_col = int(dust_hash-2)
            # relative collision velocity and angle
            planet_xyz = np.array(ps['Planet'].xyz)
            planet_vxyz = np.array(ps['Planet'].vxyz)
            dust_xyz = np.array(ps_dust_col.xyz)
            dust_vxyz = np.array(ps_dust_col.vxyz)
            vrel, mu = compute_collision_info(planet_xyz, planet_vxyz, dust_xyz, dust_vxyz, R_Planet)
            vrel_f[ind_col] = vrel / 1e3  # km/s
            mu_f[ind_col] = mu           
            j = 2 if hash_p1 == hash_Planet else 1 # remove dust
            print(f'Hit the Planet: dust {dust_hash} at {sim.t/yr:.5f} [yr]')
        
        CJ_final[ind_col] = get_jacobi_const(sim, ps_dust_col)
        x_p_f_xyz[ind_col], v_p_f_xyz[ind_col] = ps['Planet'].xyz, ps['Planet'].vxyz # planet final position and velocity
        x_d_f_xyz[ind_col], v_d_f_xyz[ind_col] = ps_dust_col.xyz, ps_dust_col.vxyz # dust final position and velocity
        final_fate[ind_col] = fate
        lifetime[ind_col] = sim.t/yr
        
        ps["Star"].m = m_Star*(1-beta)
        a_d_f[ind_col], e_d_f[ind_col] = ps_dust_col.orbit(primary=ps['Star']).a, ps_dust_col.orbit(primary=ps['Star']).e
        ps["Star"].m = m_Star
        
        return j                            
    
    sim.collision = "direct"
    sim.collision_resolve = collision_function

    
    # start integration
    Noutput = 200
    times = np.linspace(0., endtime, Noutput)

    
    CJ_array, a_d_array, e_d_array = np.zeros((N_dust, Noutput)), np.zeros((N_dust, Noutput)), np.zeros((N_dust, Noutput))

    for i, time in enumerate(times):
        sim.integrate(time)

        if sim.N == 2:
            print ('No dusts left. Finish integration. :)')
            break
        
        for l in range(2, sim.N):
            hash_ps_l = ps[l].hash.value
            ind_ps_l = hash_ps_l - 2
            CJ_array[ind_ps_l, i] = get_jacobi_const(sim, ps[l])
            ps["Star"].m = m_Star*(1-beta)
            a_d_array[ind_ps_l, i], e_d_array[ind_ps_l, i] = ps[l].orbit(primary=ps['Star']).a, ps[l].orbit(primary=ps['Star']).e
            ps["Star"].m = m_Star
            
        # ejection judgement
        E_ps_list = []
        distance_pd_list = []
        for j in range(2, sim.N):
            E_ps_list.append(get_E(sim, ps[j]))
            distance_pd_list.append(np.linalg.norm(np.array(ps[j].xyz)-np.array(ps['Planet'].xyz)))#  

        index_ej = np.where((np.array(E_ps_list)>0) & (np.array(distance_pd_list)>10*a_Planet))[0] + 2

        l = 0 # count of dusts already removed in this round
        for k in range(len(index_ej)):
            hash_ej = ps[int(index_ej[k])-l].hash.value
            ps_dust_ej = find_particle_by_hash(sim, hash_ej)
            ind_ej = int(hash_ej-2)
            
            CJ_final[ind_ej] = get_jacobi_const(sim, ps_dust_ej)
            x_p_f_xyz[ind_ej], v_p_f_xyz[ind_ej] = ps['Planet'].xyz, ps['Planet'].vxyz # planet final position and velocity
            x_d_f_xyz[ind_ej], v_d_f_xyz[ind_ej] = ps_dust_ej.xyz, ps_dust_ej.vxyz # dust final position and velocity
            final_fate[ind_ej] = 'ejection'
            lifetime[ind_ej] = sim.t/yr

            ps["Star"].m = m_Star*(1-beta)
            a_d_f[ind_ej], e_d_f[ind_ej] = ps_dust_ej.orbit(primary=ps['Star']).a, ps_dust_ej.orbit(primary=ps['Star']).e
            ps["Star"].m = m_Star
            
            print(f'Ejection: dust {hash_ej} at {sim.t/yr:.5f} [yr]')            
            sim.remove(int(index_ej[k])-l)
            l += 1
            
        
    # incomplete
    if sim.N != 2:
        for j in range(2, sim.N):
            hash_inc = ps[j].hash.value
            ps_dust_inc = find_particle_by_hash(sim, hash_inc)
            ind_inc = int(hash_inc-2)
            
            CJ_final[ind_inc] = get_jacobi_const(sim, ps_dust_inc)
            x_p_f_xyz[ind_inc], v_p_f_xyz[ind_inc] = ps['Planet'].xyz, ps['Planet'].vxyz # planet final position and velocity
            x_d_f_xyz[ind_inc], v_d_f_xyz[ind_inc] = ps_dust_inc.xyz, ps_dust_inc.vxyz # dust final position and velocity
            final_fate[ind_inc] = 'incomplete'
            lifetime[ind_inc] = sim.t/yr

            ps["Star"].m = m_Star*(1-beta)
            a_d_f[ind_inc], e_d_f[ind_inc] = ps_dust_inc.orbit(primary=ps['Star']).a, ps_dust_inc.orbit(primary=ps['Star']).e
            ps["Star"].m = m_Star
            print(f'Incomplete: dust {hash_inc} at {sim.t/yr:.5f} [yr]')

         
        
    # outcome
    data = {
    "m_Star/m_Sun": k1*np.ones(N_dust),
    "m_Planet/m_J": k2*np.ones(N_dust),
    "a_p/R_Sun": k_ap*np.ones(N_dust),
    "R_sub/R_Sun": R_sub_To_R_Sun*np.ones(N_dust),
    "R_Planet/R_Sun": R_Planet/R_Sun*np.ones(N_dust),
    "beta": beta*np.ones(N_dust),
    "a_d_i/a_p": a_initial/a_Planet*np.ones(N_dust),
    'hash': hash_array,
    'a_initial': a_initial_array,
    'e_initial': e_initial_array,
    'inc_initial': inc_initial_array,
    'Omega_initial': Omega_initial_array,
    'pomega_initial': pomega_initial_array,
    'M_initial': M_initial_array,
    'CJ_array': [CJ_array[l].tolist() for l in range(CJ_array.shape[0])],
    'a_d_array': [a_d_array[l].tolist() for l in range(a_d_array.shape[0])],
    'e_d_array': [e_d_array[l].tolist() for l in range(e_d_array.shape[0])],
    'CJ_final': CJ_final,
    'x_p_f_xyz': [vec for vec in x_p_f_xyz], # flatten: a full [x, y, z] vector as a list
    'v_p_f_xyz': [vec for vec in v_p_f_xyz],
    'x_d_f_xyz': [vec for vec in x_d_f_xyz],
    'v_d_f_xyz': [vec for vec in v_d_f_xyz],
    'final_fate': final_fate,
    'lifetime': lifetime,
    'a_d_f': a_d_f,
    'e_d_f': e_d_f,
    "vrel_f[km/s]": vrel_f,
    "mu":mu_f
    }
    
    df = pd.DataFrame(data)
    
    return df
