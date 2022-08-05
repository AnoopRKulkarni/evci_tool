#!/usr/bin/env python
# coding: utf-8

#
# MPEN EVCI Tool
# Authors: Anoop R Kulkarni
# Version 1.0
# Jun 24, 2022

#@title Import libraries
import argparse
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

import warnings
warnings.filterwarnings("ignore")

from evci_config import *
from evci_model import *

def run_episode(s_df,txt,OUTPUT_PATH):
   print('\n' + txt.capitalize() + ' Analysis')
   print('________________\n')
   total = s_df.shape[0]
   s_df = s_df[s_df['year 1'] == 1]
   s_df = s_df.reset_index(drop=True)
   Nc = s_df.shape[0]
   print(f'Number of sites: {Nc}/{total}')
   
   #@title Compute scores
   
   backoff_factor = 2 #@param {type:"slider", min:1, max:5, step:1}
   
   u_df = run_analysis(args,s_df,Nc,backoff_factor=backoff_factor)
   
   print(f'Total capex charges = INR Cr {sum(u_df.capex)/1e7:.2f}')
   print(f'Total opex charges = INR Cr {sum(u_df.opex)/1e7:.2f}')
   print(f'Total Margin = INR Cr {sum(u_df.margin)/1e7:.2f}')        
   
   #@title Prepare data
   s_u_df = s_df.copy()
   
   s_u_df['utilization'] = u_df.utilization
   s_u_df['unserviced'] = u_df.unserviced
   s_u_df['capex'] = u_df.capex
   s_u_df['opex'] = u_df.opex
   s_u_df['margin'] = u_df.margin
   s_u_df['max vehicles'] = u_df['max vehicles']
   s_u_df['estimated vehicles'] = u_df['estimated vehicles']
   
   #@title Save initial analysis to Excel
   output_df = s_u_df.copy()
   output_df.drop('geometry', axis=1, inplace=True)
   output_df.to_excel(OUTPUT_PATH + args.prefix + '_evci_analysis_'+txt+'_'+args.expt+'.xlsx')
   return s_u_df


def main(args):
   #@title set paths
   EXPT = args.expt

   INPUT_PATH = args.input_dir + '/' + args.expt + '/'
   OUTPUT_PATH = args.output_dir + '/' + args.expt + '/'

   #@title Read Master Data Excel Book
   newdata = pd.read_excel(INPUT_PATH + args.prefix + '/' + args.master_xlsx, sheet_name=None)
   datasheets = list(newdata.keys())
   print(f'\nDatasheets: {datasheets}')

   #@title Read required data sheets only
   df = gpd.read_file(INPUT_PATH + args.prefix + '/' + args.gis)
   
   data = {}
   
   for s in datasheets:
     data[s] = newdata[s]
     data[s]['Name'] = data[s]['Name']
     data[s]['Sheet'] = s
     data[s]['Latitude'] = pd.to_numeric(data[s]['Latitude'])
     data[s]['Longitude'] = pd.to_numeric(data[s]['Longitude'])
     data[s]['geometry'] = [shapely.geometry.Point(xy) for xy in 
                            zip(data[s]['Longitude'], 
                                data[s]['Latitude'])]
   data_df = {}
   
   for s in data:
     data_df[s] = gpd.GeoDataFrame(data[s], 
                                   geometry=data[s]['geometry'])
   
   #@title Prepare data
   s_df = pd.DataFrame(columns=['Name', 'Sheet',
                                'Latitude', 'Longitude',
                                'Traffic congestion',
                                'year 1',
                                'kiosk hoarding',
                                'hoarding margin',
                                'geometry'])
   
   for s in datasheets:
     s_df = s_df.reset_index(drop=True)
     print(s,data_df[s].shape[0])
     for i in range(data_df[s].shape[0]):
         s_df.loc[i+s_df.shape[0]] = [
               data_df[s].loc[i].Name, 
               s,
               data_df[s].loc[i].Latitude, 
               data_df[s].loc[i].Longitude, 
               data_df[s].loc[i]['Traffic congestion'],
               data_df[s].loc[i]['Year for Site recommendation'],
               data_df[s].loc[i]['Hoarding/Kiosk (1 is yes & 0 is no)'],
               data_df[s].loc[i]['Hoarding Margin (30% for year 1 and 15% for year 2 of base cost 9 lakhs)'],
               data_df[s].loc[i].geometry
           ]
   
   s_u_df = run_episode(s_df,'initial',OUTPUT_PATH)

   #@title Threshold and cluster
   if args.threshold_and_cluster:
     clustering_candidates = s_u_df[(s_u_df.utilization <= 0.2) & (s_u_df['year 1'] == 1)]
     points = np.array((clustering_candidates.apply(lambda x: list([x['Latitude'], x['Longitude']]),axis=1)).tolist())
     Z = linkage (points, method='complete', metric='euclidean');
     plt.figure(figsize=(14,8))
     dendrogram(Z);
     max_d = 0.01
     clusters = fcluster(Z, t=max_d, criterion='distance')
     clustered_candidates = gpd.GeoDataFrame(clustering_candidates)
     #base = grid_df.plot(color='none', alpha=0.2, edgecolor='black', figsize=(8,8))
     #clustered_candidates.plot(ax=base, column=clusters, legend=True)

   #@title Build final list of sites
   confirmed_sites = s_u_df[s_u_df.utilization > 0.2]
   if args.threshold_and_cluster:
     val, ind = np.unique (clusters, return_index=True)
     clustered_sites = clustered_candidates.reset_index(drop=True)
     clustered_sites = clustered_sites.iloc[clustered_sites.index.isin(ind)]
     final_list_of_sites = pd.concat([confirmed_sites, clustered_sites], axis=0)
   else:
     final_list_of_sites = confirmed_sites.copy()

   if args.threshold_and_cluster:
     s_u_df = run_episode(final_list_of_sites,'cluster',OUTPUT_PATH)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir", type=str, default='input',
        help="""this is an optional argument. 'input/' assumed"""
    )
    parser.add_argument(
        "--output_dir", type=str, default='output',
        help="""this is an optional argument. 'output/' assumed"""
    )
    parser.add_argument(
        "--expt", type=str, default=None,
        help="""name of the experiment for unique output file names"""
    )
    parser.add_argument(
        "--threshold_and_cluster", type=bool, default=False,
        choices = [True, False],
        help="""the input list of sites will be clustered and thresholded"""
    )
    args = parser.parse_args()
   
    main(args)

