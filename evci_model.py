#@title Define analytical model
# Functions of the formulation to compute score, capex, opex and margin

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import shapely

import os
import pandas as pd
from tqdm import tqdm

import ipywidgets as widgets

import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.vq import kmeans2, whiten
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster

M = {'2W', '4WS', '4WF'}

Dj = {'2W': 2.2, '4WS': 22, '4WF': 60}
Hj = {'2W': 5, '4WS': 15, '4WF': 15}
Qj = {'2W': 1250, '4WS': 9143, '4WF': 18286}
tj = {'2W': 1, '4WS': 1.5, '4WF': 0.5}
Mj = {'2W': 500, '4WS': 500, '4WF': 500}

Gk = {1: 1.0, 2: 1.0, 3: 1.0}
N = 500
Ng = 0

# peak vehicles through crowded junctions in a day ~ 1.5L

peak_traffic = [
         4826, 4826, 5228, 5228, 5228, 5630, 6434, 6836, 6836, 
         6434, 6032, 6032, 6032, 6032, 6434, 6836, 7239, 8043, 
         8043, 8043, 6836, 6032, 5630, 5228       
]

def run_analysis(args,s_df,Nc,backoff_factor=1):

  C = set(args.choose_chargers)
   
  Gi = [0]*Nc
  di = [0]*Nc
  Wi = [0]*Nc
  Ri = [0]*Nc
  Ai = [50]*Nc
  Li = [1500]*Nc
  Bi = [0.25 * 3.5 * 24 * 365]*Nc # 25% of Rs 3.5/KWh per year
   
  CH = [args.hoarding_cost]*Nc
  CK = [args.kiosk_cost]*Nc
   
  MH = [s_df.loc[i]['hoarding margin'] for i in range(Nc)]
  MK = [0.15]*Nc

  Kj = {'2W': args.capex_2W, '4WS': args.capex_4W_slow_charging, '4WF': args.capex_4W_fast_charging}
 
  timeslots = {k: 24/v for k, v in tj.items()}
 
  Eg = {k: [5.5] * int(v) for k, v in timeslots.items()}
  Er = {k: [0] * int(v) for k, v in timeslots.items()}
  Mg = {k: [5.5 * 0.15] * int(v) for k, v in timeslots.items()}
  Mr = {k: [0] * int(v) for k, v in timeslots.items()}
  l  = {k: [1] * int(v) for k, v in timeslots.items()}
 
  pj = {1: args.year1_conversion,
         2: args.year2_conversion,
         3: args.year3_conversion}
 
  Pj = max(pj.values())
 
  # Average traffic approx 80% of peak
  avg_traffic = [i*.8 for i in peak_traffic]
  # 2W and 4W assumed to be 60% and 20% respectively
  avg_traffic_2W = [i*.6 for i in avg_traffic]
  avg_traffic_4W = [i*.2 for i in avg_traffic]
  djworking_hourly_2W = [i/5 for i in avg_traffic_2W]
  djworking_hourly = [i/5 for i in avg_traffic_4W]
  djworking_half_hourly = [val for val in djworking_hourly
                          for _ in (0, 1)]
  djworking_one_and_half_hourly = list(np.mean(np.array(djworking_half_hourly).reshape(-1, 3), axis=1))
 
  djworking = {}
  djworking['2W'] = [np.round(i,0) for i in djworking_hourly_2W]
  djworking['4WF'] = [np.round(i,0) for i in djworking_half_hourly]
  djworking['4WS'] = [np.round(i,0) for i in djworking_one_and_half_hourly]
 
  djholiday = {}
  djholiday['2W'] = [np.round(i*args.holiday_percentage,0) for i in djworking_hourly_2W]
  djholiday['4WF'] = [np.round(i*args.holiday_percentage,0) for i in djworking_half_hourly]
  djholiday['4WS'] = [np.round(i*args.holiday_percentage,0) for i in djworking_one_and_half_hourly]
 
  qjworking = {'4WS': [args.slow_charging] * int(timeslots['4WS']),
              '4WF': [args.fast_charging] * int(timeslots['4WF']),
              '2W' : [args.fast_charging + args.slow_charging] * int(timeslots['2W']), }
  qjholiday = {'4WS': [args.slow_charging] * int(timeslots['4WS']),
              '4WF': [args.fast_charging] * int(timeslots['4WF']),
              '2W' : [args.fast_charging + args.slow_charging] * int(timeslots['2W']), }
 
  Cij = {'2W': [4]*Nc, '4WS': [1]*Nc, '4WF':[1]*Nc}
 

  def score(j,i,hj,k,backoff=True, backoff_factor=1):
      distance_from_i = s_df_distances[s_df_distances > 0][i].sort_values()/1e3
      closer_to_i = distance_from_i[distance_from_i <= 5.0]
      try:
        congestion = float(s_df.loc[i]['Traffic congestion'])
      except:
        congestion = 1.0
      
      nw = qjworking[j][hj] * djworking[j][hj] * pj[k] * congestion
      nh = qjholiday[j][hj] * djholiday[j][hj] * pj[k] * congestion
      
      if backoff:
          for el in closer_to_i:
              nw *= (1 - np.exp(-el*backoff_factor))
              nh *= (1 - np.exp(-el*backoff_factor))
      
      tw = th = 0
      if (Cij[j][i] > 0): tw = nw * (tj[j]/Cij[j][i])
      if (Cij[j][i] > 0): th = nh * (tj[j]/Cij[j][i])
      
      uw = uh = tj[j]
      if (tw <= tj[j]): uw = tw 
      if (th <= tj[j]): uh = th
          
      vw = vh = 0
      if (tw > tj[j]): vw = (tw - tj[j]) * (Cij[j][i]/tj[j])
      if (th > tj[j]): vh = (th - tj[j]) * (Cij[j][i]/tj[j])
      
      norm_uw, norm_uh = uw/tj[j], uh/tj[j]
      norm_vw = vw/nw
      norm_vh = vh/nh
      
      return norm_uw, norm_uh, norm_vw, norm_vh
  
  def capex(i):
      retval = 0
      for j in C:
          retval += Cij[j][i]*Kj[j] + Wi[i] * di[i] * Cij[j][i]
      return retval
  
  def opex(i):
      op_e = 0
      op_l = 0
  
      for k in Gk:
          for j in C:
              for h in range(int(timeslots[j])):
                  sw, sh, _, _ = score (j,i,h,k)
                  op_e += 300 * Cij[j][i] * sw * tj[j] * Dj[j] * (l[j][h] * Eg[j][h] + (1-l[j][h]) * Er[j][h])
                  op_e +=  65 * Cij[j][i] * sh * tj[j] * Dj[j] * (l[j][h] * Eg[j][h] + (1-l[j][h]) * Er[j][h])
      op_l = Li[i] * Ai[i] + CH[i] + CK[i]
      return op_e + op_l
      
  def margin(i):
      margin_e = 0
      margin_l = 0
  
      for k in Gk:
        for j in C:
            for h in range(int(timeslots[j])):
                sw, sh, _, _ = score (j,i,h,k)
                margin_e += 300 * Cij[j][i] * sw * tj[j] * Dj[j] * (l[j][h] * Mg[j][h] + (1-l[j][h]) * Mr[j][h])
                margin_e +=  65 * Cij[j][i] * sh * tj[j] * Dj[j] * (l[j][h] * Mg[j][h] + (1-l[j][h]) * Mr[j][h])
      margin_l = Bi[i] * Ai[i] + MH[i] + MK[i]
      return margin_e + margin_l
  
  global s_df_distances
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

  for i in tqdm(range(Nc)):
    max_vehicles = 0
    for j in C:
      max_vehicles += timeslots[j]*Cij[j][i]
    op_e = 0
    op_l = 0
    margin_e = 0
    margin_l = 0
    year_u_avg = np.array([])
    year_v_avg = np.array([])
    for k in Gk:
      chargertype_u_avg = np.array([])
      chargertype_v_avg = np.array([])
      for j in C:
        uw_day_avg = np.array([])
        uh_day_avg = np.array([])
        vw_day_avg = np.array([])
        vh_day_avg = np.array([])              
        for h in range(int(timeslots[j])):
          uw, uh, vw, vh = score (j,i,h,k, backoff_factor=backoff_factor)
          uw_day_avg = np.append(uw_day_avg, uw)
          uh_day_avg = np.append(uh_day_avg, uh)
          vw_day_avg = np.append(vw_day_avg, vw)
          vh_day_avg = np.append(vh_day_avg, vh)                  
          op_e += 300 * Cij[j][i] * uw * tj[j] * Dj[j] * (l[j][h] * Eg[j][h] + (1-l[j][h]) * Er[j][h])
          op_e +=  65 * Cij[j][i] * uh * tj[j] * Dj[j] * (l[j][h] * Eg[j][h] + (1-l[j][h]) * Er[j][h])            
          margin_e += 300 * Cij[j][i] * uw * tj[j] * Dj[j] * (l[j][h] * Mg[j][h] + (1-l[j][h]) * Mr[j][h])
          margin_e +=  65 * Cij[j][i] * uh * tj[j] * Dj[j] * (l[j][h] * Mg[j][h] + (1-l[j][h]) * Mr[j][h])        
        weighted_u = (300.0*uw_day_avg.mean() + 65.0*uh_day_avg.mean()) / 365.0
        weighted_v = (300.0*vw_day_avg.mean() + 65.0*vh_day_avg.mean()) / 365.0
        chargertype_u_avg = np.append(chargertype_u_avg, weighted_u)
        chargertype_v_avg = np.append(chargertype_v_avg, weighted_v)
      year_u_avg = np.append(year_u_avg, chargertype_u_avg.mean())
      year_v_avg = np.append(year_v_avg, chargertype_v_avg.mean())
      op_l += Li[i] * Ai[i] + CH[i] + CK[i]
      margin_l += Bi[i] * Ai[i] + MH[i] + MK[i]
    site_capex = capex(i)
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
