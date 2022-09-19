# AUTOGENERATED! DO NOT EDIT! File to edit: ../01_model.ipynb.

# %% auto 0
__all__ = ['score', 'capex', 'opex', 'margin', 'run_analysis']

# %% ../01_model.ipynb 3
import numpy as np
import pandas as pd
import geopandas as gpd

import shapely

import os
from tqdm import tqdm

from .config import *

import warnings
warnings.filterwarnings("ignore")

# %% ../01_model.ipynb 4
def score(r,s_df_distances,j,i,hj,k,backoff=True, backoff_factor=1):
    "This function computes the utilization score of each site."
    
    distance_from_i = s_df_distances[s_df_distances > 0][i].sort_values()/1e3
    closer_to_i = distance_from_i[distance_from_i <= 5.0]
    try:
        congestion = float(s_df_distances.loc[i]['Traffic congestion'])
    except:
        congestion = 1.0

    nw = r['qjworking'][j][hj] * r['djworking'][j][hj] * r['pj'][k] * congestion
    nh = r['qjholiday'][j][hj] * r['djholiday'][j][hj] * r['pj'][k] * congestion

    if backoff:
        for el in closer_to_i:
            nw *= (1 - np.exp(-el*backoff_factor))
            nh *= (1 - np.exp(-el*backoff_factor))

    tw = th = 0
    if (r['Cij'][j][i] > 0): tw = nw * (r['tj'][j]/r['Cij'][j][i])
    if (r['Cij'][j][i] > 0): th = nh * (r['tj'][j]/r['Cij'][j][i])

    uw = uh = r['tj'][j]
    if (tw <= r['tj'][j]): uw = tw 
    if (th <= r['tj'][j]): uh = th

    vw = vh = 0
    if (tw > r['tj'][j]): vw = (tw - r['tj'][j]) * (r['Cij'][j][i]/r['tj'][j])
    if (th > r['tj'][j]): vh = (th - r['tj'][j]) * (r['Cij'][j][i]/r['tj'][j])

    norm_uw, norm_uh = uw/r['tj'][j], uh/r['tj'][j]
    
    if nw>0: norm_vw = vw/nw
    else: norm_vw = 0
    if nh>0: norm_vh = vh/nh
    else: norm_vh = 0

    return norm_uw, norm_uh, norm_vw, norm_vh

# %% ../01_model.ipynb 6
def capex(r,i):
    "This function computes the capex requirements of each site"
    retval = 0
    for j in r['C']:
        retval += r['Cij'][j][i]*r['Kj'][j] + r['Wi'][i] * r['di'][i] * r['Cij'][j][i]
    return retval

# %% ../01_model.ipynb 8
def opex(r,s_df_distances,i):
    "This function computes the opex for each site."
    op_e = 0
    op_l = 0

    for k in r['years_of_analysis']:
        for j in r['C']:
            for h in range(int(r['timeslots'][j])):
                sw, sh, _, _ = score (r,s_df_distances,j,i,h,k)
                op_e += 300 * r['Cij'][j][i] * sw * r['tj'][j] * r['Dj'][j] * (r['l'][j][h] * r['Eg'][j][h] + (1-r['l'][j][h]) * r['Er'][j][h])
                op_e +=  65 * r['Cij'][j][i] * sh * r['tj'][j] * r['Dj'][j] * (r['l'][j][h] * r['Eg'][j][h] + (1-r['l'][j][h]) * r['Er'][j][h])
    op_l = r['Li'][i] * r['Ai'][i] + r['CH'][i] + r['CK'][i]
    return op_e + op_l

# %% ../01_model.ipynb 10
def margin(r,s_df_distances,i):
    "This function computes the margins per site."
    margin_e = 0
    margin_l = 0

    for k in r['years_of_analysis']:
        for j in r['C']:
            for h in range(int(r['timeslots'][j])):
                sw, sh, _, _ = score (r,s_df_distances,j,i,h,k)
                margin_e += 300 * r['Cij'][j][i] * sw * r['tj'][j] * r['Dj'][j] * (r['l'][j][h] * r['Mg'][j][h] + (1-r['l'][j][h]) * r['Mr'][j][h])
                margin_e +=  65 * r['Cij'][j][i] * sh * r['tj'][j] * r['Dj'][j] * (r['l'][j][h] * r['Mg'][j][h] + (1-r['l'][j][h]) * r['Mr'][j][h])
    margin_l = r['Bi'][i] * r['Ai'][i] + r['MH'][i] + r['MK'][i]
    return margin_e + margin_l

# %% ../01_model.ipynb 12
def run_analysis(m,s,t,g,ui_inputs,s_df,backoff_factor=1):
    "This function runs analysis for a given set of sites."

    r = read_globals(m,s,t,g,ui_inputs)
    
    u_df = pd.DataFrame(columns=['utilization', 
                             'unserviced', 
                             'capex', 
                             'opex', 
                             'margin', 
                             'max vehicles', 
                             'estimated vehicles'
                             ])    
    s_df_crs = gpd.GeoDataFrame(s_df, crs='EPSG:4326')
    s_df_crs = s_df_crs.to_crs('EPSG:5234')
    s_df_distances = s_df_crs.geometry.apply(lambda g: s_df_crs.distance(g))      
    
    s_df_distances['Traffic congestion'] = s['sites']['Traffic congestion']
    
    Nc = s_df.shape[0]
    
    for i in tqdm(range(Nc)):
        max_vehicles = 0
        # run through selected charger types
        for j in r['M']:
            max_vehicles += r['timeslots'][j]*r['Cij'][j][i]
        max_vehicles = int(np.round(max_vehicles,0))
        op_e = 0
        op_l = 0
        margin_e = 0
        margin_l = 0
        year_u_avg = np.array([])
        year_v_avg = np.array([])
        for k in r['years_of_analysis']:
            chargertype_u_avg = np.array([])
            chargertype_v_avg = np.array([])
            for j in r['C']:
                uw_day_avg = np.array([])
                uh_day_avg = np.array([])
                vw_day_avg = np.array([])
                vh_day_avg = np.array([])              
                for h in range(int(r['timeslots'][j])):
                    uw, uh, vw, vh = score (r,s_df_distances,j,i,h,k,backoff_factor=backoff_factor)
                    uw_day_avg = np.append(uw_day_avg, uw)
                    uh_day_avg = np.append(uh_day_avg, uh)
                    vw_day_avg = np.append(vw_day_avg, vw)
                    vh_day_avg = np.append(vh_day_avg, vh)                  
                    op_e += 300 * r['Cij'][j][i] * uw * r['tj'][j] * r['Dj'][j] * (r['l'][j][h] * r['Eg'][j][h] + (1-r['l'][j][h]) * r['Er'][j][h])
                    op_e +=  65 * r['Cij'][j][i] * uh * r['tj'][j] * r['Dj'][j] * (r['l'][j][h] * r['Eg'][j][h] + (1-r['l'][j][h]) * r['Er'][j][h])            
                    margin_e += 300 * r['Cij'][j][i] * uw * r['tj'][j] * r['Dj'][j] * (r['l'][j][h] * r['Mg'][j][h] + (1-r['l'][j][h]) * r['Mr'][j][h])
                    margin_e +=  65 * r['Cij'][j][i] * uh * r['tj'][j] * r['Dj'][j] * (r['l'][j][h] * r['Mg'][j][h] + (1-r['l'][j][h]) * r['Mr'][j][h])        
                weighted_u = (300.0*uw_day_avg.mean() + 65.0*uh_day_avg.mean()) / 365.0
                weighted_v = (300.0*vw_day_avg.mean() + 65.0*vh_day_avg.mean()) / 365.0
                chargertype_u_avg = np.append(chargertype_u_avg, weighted_u)
                chargertype_v_avg = np.append(chargertype_v_avg, weighted_v)
            year_u_avg = np.append(year_u_avg, chargertype_u_avg.mean())
            year_v_avg = np.append(year_v_avg, chargertype_v_avg.mean())
            op_l += r['Li'][i] * r['Ai'][i] + r['CH'][i] + r['CK'][i]
            margin_l += r['Bi'][i] * r['Ai'][i] + r['MH'][i] + r['MK'][i]
        site_capex = capex(r,i)
        estimated_vehicles = np.round(year_u_avg.mean()*max_vehicles,0)
        u_df.loc[i] = [ year_u_avg.mean(), 
                       year_v_avg.mean(),
                       site_capex,
                       op_e + op_l,
                       margin_e + margin_l,
                       max_vehicles,
                       estimated_vehicles
                       ]
    return u_df
