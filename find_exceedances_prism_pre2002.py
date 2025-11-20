#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import xarray as xr
import numpy as np

from scipy.ndimage import label, gaussian_filter
from scipy.stats import pearsonr
from sklearn.cluster import DBSCAN

import os
import sys

from ari_exceedance_map_functions import *
from qc_functions import *

dataset = "prism"

#duration = 72 
duration = int(sys.argv[1])
#year = 2018
year = int(sys.argv[2])

lat_rad = 2.5 ### how big of a box to plot? 
lon_rad = 3.25 ### These are how many degrees on either side of the center point


# In[3]:


clevs, cmap, norm = precip_colormap()

### read in precip zarrs
prism = xr.open_zarr("/glade/campaign/univ/ucsu0118/gridded_precip_zarr/prism_24h.zarr/")

### and ARI grid
ari10_grid = xr.open_dataset("/glade/campaign/univ/ucsu0118/atlas14/allusa_ari_10yr_"+str(duration)+"hr_xarray_prismgrid.nc")
ari100_grid = xr.open_dataset("/glade/campaign/univ/ucsu0118/atlas14/allusa_ari_100yr_"+str(duration)+"hr_xarray_prismgrid.nc")
ari1000_grid = xr.open_dataset("/glade/campaign/univ/ucsu0118/atlas14/allusa_ari_1000yr_"+str(duration)+"hr_xarray_prismgrid.nc")

### set the lat/lon in 'data' to exactly match the ARI grid file
prism['lon'] = ari10_grid['lon']
prism['lat'] = ari10_grid['lat']

# In[4]:

print(year)
date_range = pd.date_range(str(year)+'-01-01 12:00',str(year)+'-12-31 23:59', freq='1d')
#date_range = pd.date_range('2018-09-15 12:00','2018-09-22 12:00')


# ### now, loop over each day in the dataset to identify any exceedances of 10/100/1000 year thresholds. If there are, do the dbscan spatial clustering and perform the QC on each 'event'. If not, then just go on to the next day!

# In[7]:


### initialize some dataframes
exceed10_df_all = pd.DataFrame()
exceed100_df_all = pd.DataFrame()
exceed1000_df_all = pd.DataFrame()

for time_pd in date_range:

    print(time_pd)

    ### get the slice for the current time
    if duration > 24:
        prism_slice = prism.sel(time=slice(time_pd - pd.Timedelta(hours=duration-1),time_pd)).tp.sum(dim='time')
    elif duration==24:
        try:
            prism_slice = prism.sel(time=time_pd).tp
        except:
            print("data missing on this date, moving on")
            continue
    elif duration < 24:
        print("subdaily not yet available")

    ### find the exceedances on this date
    prism_exceed10 = prism_slice.where((prism_slice > ari10_grid.precip).compute(), drop=True).to_dataset()
    prism_exceed100 = prism_slice.where((prism_slice > ari100_grid.precip).compute(), drop=True).to_dataset()    
    prism_exceed1000 = prism_slice.where((prism_slice > ari1000_grid.precip).compute(), drop=True).to_dataset()

    ### get 10-year exceedances into a dataframe
    if (prism_exceed10.tp.shape[0] > 0): ### if there are events
        prism_exceed10_df = create_exceed_df(prism_exceed10,ari10_grid,time_pd)

    ### repeat for 100
    if (prism_exceed100.tp.shape[0] > 0): ### if there are events
        prism_exceed100_df = create_exceed_df(prism_exceed100,ari100_grid,time_pd)
    else:
        prism_exceed100_df = None

    ### repeat for 1000
    if (prism_exceed1000.tp.shape[0] > 0): ### if there are events
        prism_exceed1000_df = create_exceed_df(prism_exceed1000,ari1000_grid,time_pd)
    else:
        prism_exceed1000_df = None

    ### now, if there are any 10-year events on this day, go on to spatially cluster them and loop through them...
    ### do dbscan clustering to find 'events' on this day
    if (prism_exceed10.tp.shape[0] > 0): ### if there are events
        df_clustered, n_clusters = cluster_points_dbscan(prism_exceed10_df, max_distance_km=350, min_points=1)
        print(str(n_clusters)+" events on this day")

        for j in range(0,n_clusters):
            print("event "+str(j+1))
        
            df_event = df_clustered[df_clustered.event_num==j]
            df_event_sort = df_event.sort_values(by='tp_pct_of_ari', ascending=False) ### sort with highest pct of ari at the top

            event_num = j
        
            this_lon = df_event_sort.iloc[0].lon
            this_lat = df_event_sort.iloc[0].lat
            
            minlon = df_event_sort.lon.min() - lon_rad
            maxlon = df_event_sort.lon.max() + lon_rad
            minlat = df_event_sort.lat.min() - lat_rad
            maxlat = df_event_sort.lat.max() + lat_rad
            
            ari10_this = ari10_grid.sel(lon=slice(minlon,maxlon),lat=slice(minlat,maxlat)).precip
            ari100_this = ari100_grid.sel(lon=slice(minlon,maxlon),lat=slice(minlat,maxlat)).precip
            ari1000_this = ari1000_grid.sel(lon=slice(minlon,maxlon),lat=slice(minlat,maxlat)).precip
                
            if duration > 24:
                prism_slice = prism.sel(time=slice(time_pd - pd.Timedelta(hours=duration-1),time_pd),lon=slice(minlon,maxlon),lat=slice(minlat,maxlat))
                prism_this = prism_slice.tp.sum(dim='time')
            elif duration==24:
                prism_this = prism.sel(time=time_pd,lon=slice(minlon,maxlon),lat=slice(minlat,maxlat)).tp
            elif duration<24:
                print("subdaily not available yet")       
 
            prism_exceed10_ds = xr.where(prism_this > ari10_this,1,np.nan)
            prism_exceed100_ds = xr.where(prism_this > ari100_this,1,np.nan)
            prism_exceed1000_ds = xr.where(prism_this > ari1000_this,1,np.nan)

            data1_exceed10 = np.isfinite(prism_exceed10_ds.values)
            data1_exceed100 = np.isfinite(prism_exceed100_ds.values)
            data1_exceed1000 = np.isfinite(prism_exceed1000_ds.values)

            lons = prism_this.lon.values
            lats = prism_this.lat.values
            
            ### add these points to the full dataframe for 10-year
            exceed10_df_all = pd.concat([exceed10_df_all,df_event])
            if prism_exceed100_df is not None:
                exceed100_df_this = parse_latlons(prism_this,data1_exceed100,prism_exceed100_df,event_num,time_pd)
                exceed100_df_all = pd.concat([exceed100_df_all,exceed100_df_this])
            if prism_exceed1000_df is not None:
                exceed1000_df_this = parse_latlons(prism_this,data1_exceed1000,prism_exceed1000_df,event_num,time_pd)
                exceed1000_df_all = pd.concat([exceed1000_df_all,exceed1000_df_this])


### for the pre-2002 years, we don't have a dataset to compare to, so no need for QC. We'll just write out the dataframes to csv. (We won't make plots here either.) 
os.system("mkdir -p "+dataset+"/auto_qc/"+str(duration).zfill(2)+"h/"+str(year)+"/") 
exceed10_df_all.reset_index(drop=True).to_csv(dataset+"/auto_qc/"+str(duration).zfill(2)+"h/"+str(year)+"/"+dataset+"_"+str(year)+"_all_points_10y"+str(duration).zfill(2)+"h.csv", index=None)
exceed100_df_all.reset_index(drop=True).to_csv(dataset+"/auto_qc/"+str(duration).zfill(2)+"h/"+str(year)+"/"+dataset+"_"+str(year)+"_all_points_100y"+str(duration).zfill(2)+"h.csv", index=None)
exceed1000_df_all.reset_index(drop=True).to_csv(dataset+"/auto_qc/"+str(duration).zfill(2)+"h/"+str(year)+"/"+dataset+"_"+str(year)+"_all_points_1000y"+str(duration).zfill(2)+"h.csv", index=None)


print("done!")




