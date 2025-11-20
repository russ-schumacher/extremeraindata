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
stage4 = xr.open_zarr("/glade/campaign/univ/ucsu0118/gridded_precip_zarr/stage4_24h_prismgrid.zarr/")
prism = xr.open_zarr("/glade/campaign/univ/ucsu0118/gridded_precip_zarr/prism_24h.zarr/")
mrms = xr.open_zarr("/glade/campaign/univ/ucsu0118/gridded_precip_zarr/mrms_24h_prismgrid.zarr/")

### pick a time that we'll use to mask out values outside conus
prism_mask = prism.sel(time='2023-06-24 12:00').tp.load()

### and ARI grid
ari10_grid = xr.open_dataset("/glade/campaign/univ/ucsu0118/atlas14/allusa_ari_10yr_"+str(duration)+"hr_xarray_prismgrid_nopnw.nc")
ari100_grid = xr.open_dataset("/glade/campaign/univ/ucsu0118/atlas14/allusa_ari_100yr_"+str(duration)+"hr_xarray_prismgrid_nopnw.nc")
ari1000_grid = xr.open_dataset("/glade/campaign/univ/ucsu0118/atlas14/allusa_ari_1000yr_"+str(duration)+"hr_xarray_prismgrid_nopnw.nc")

### set the lat/lon in 'data' to exactly match the ARI grid file
mrms['lon'] = ari10_grid['lon']
mrms['lat'] = ari10_grid['lat']

prism['lon'] = ari10_grid['lon']
prism['lat'] = ari10_grid['lat']

stage4['lon'] = ari10_grid['lon']
stage4['lat'] = ari10_grid['lat']



print(year)
date_range = pd.date_range(str(year)+'-01-01 12:00',str(year)+'-12-31 23:59', freq='1d')
#date_range = pd.date_range('2018-09-15 12:00','2018-09-22 12:00')


# ### now, loop over each day in the dataset to identify any exceedances of 10/100/1000 year thresholds. If there are, do the dbscan spatial clustering and perform the QC on each 'event'. If not, then just go on to the next day!

# In[7]:


### initialize some dataframes
exceed10_df_all = pd.DataFrame()
exceed100_df_all = pd.DataFrame()
exceed1000_df_all = pd.DataFrame()
removed10_df_all = pd.DataFrame()
removed100_df_all = pd.DataFrame()
removed1000_df_all = pd.DataFrame()

for time_pd in date_range:

    print(time_pd)

    ### get the slice for the current time
    #### change the variable names here to the ones for the dataset being evaluated!
    if duration > 24:
        #stage4_slice = stage4.sel(time=slice(time_pd - pd.Timedelta(hours=duration-1),time_pd)).tp.sum(dim='time').where(prism_mask>=0)
        #mrms_slice = mrms.sel(time=slice(time_pd - pd.Timedelta(hours=duration-1),time_pd)).tp.sum(dim='time').where(prism_mask>=0)
        prism_slice = prism.sel(time=slice(time_pd - pd.Timedelta(hours=duration-1),time_pd)).tp.sum(dim='time')
    elif duration==24:
        try:
        #    mrms_slice = mrms.sel(time=time_pd).tp.where(prism_mask>=0)
        #    stage4_slice = stage4.sel(time=time_pd).tp.where(prism_mask>=0)
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
                stage4_slice = stage4.sel(time=slice(time_pd - pd.Timedelta(hours=duration-1),time_pd),lon=slice(minlon,maxlon),lat=slice(minlat,maxlat))
                stage4_this = stage4_slice.tp.sum(dim='time').where(prism_mask>=0)
            elif duration==24:
                stage4_this = stage4.sel(time=time_pd,lon=slice(minlon,maxlon),lat=slice(minlat,maxlat)).tp.where(prism_mask>=0)
            elif duration<24:
                print("subdaily not available yet")       
 
            stage4_exceed10_ds = xr.where(stage4_this > ari10_this,1,np.nan)
            stage4_exceed100_ds = xr.where(stage4_this > ari100_this,1,np.nan)
            stage4_exceed1000_ds = xr.where(stage4_this > ari1000_this,1,np.nan)
            
            #if duration > 24:
            #    mrms_slice = mrms.sel(time=slice(time_pd - pd.Timedelta(hours=duration-1),time_pd),lon=slice(minlon,maxlon),lat=slice(minlat,maxlat))
            #    mrms_this = mrms_slice.tp.sum(dim='time').where(prism_mask>=0)
            #elif duration==24:
            #    mrms_this = mrms.sel(time=time_pd,lon=slice(minlon,maxlon),lat=slice(minlat,maxlat)).tp.where(prism_mask>=0)
            #elif duration<24:
            #    print("subdaily not available yet")

            #mrms_exceed10_ds = xr.where(mrms_this > ari10_this,1,np.nan)
            #mrms_exceed100_ds = xr.where(mrms_this > ari100_this,1,np.nan)
            #mrms_exceed1000_ds = xr.where(mrms_this > ari1000_this,1,np.nan)
            
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
            
            # Load your two datasets
            #### change data1  (this is where to set which datasets you want to use. data1 is the one you're evaluating, data2 is the one used for comparison, typically prism)
            data1_ds = prism_this
            data1 = data1_ds.values
            data2_ds = stage4_this
            data2 = data2_ds.values
    
            data1_name = "PRISM"
            data2_name = "Stage IV"
     
            #### need to change these definitions to be consistent with data1 (i.e., if data1 is mrms, make these mrms_exceed, etc.) 
            data1_exceed10 = np.isfinite(prism_exceed10_ds.values)
            data2_exceed10 = np.isfinite(stage4_exceed10_ds.values)
            data1_exceed100 = np.isfinite(prism_exceed100_ds.values)
            data2_exceed100 = np.isfinite(stage4_exceed100_ds.values)
            data1_exceed1000 = np.isfinite(prism_exceed1000_ds.values)
            data2_exceed1000 = np.isfinite(stage4_exceed1000_ds.values)
            
            lons = data1_ds.lon.values
            lats = data1_ds.lat.values
                
            # Run comprehensive comparison
            results = comprehensive_dataset_comparison(data1, data2, 
                                                           data1_exceed10, data2_exceed10, 
                                                           data1_exceed100, data2_exceed100, 
                                                           data1_exceed1000, data2_exceed1000, 
                                                           data1_name, data2_name)
                
            # Visualize results
            plot_comparison_results(data1, data2, 
                                        data1_exceed10, data2_exceed10, 
                                        data1_exceed100, data2_exceed100, 
                                        data1_exceed1000, data2_exceed1000, 
                                        lons, lats, this_lon, this_lat,
                                        minlon, maxlon, minlat, maxlat, cmap, norm,
                                        dataset, data1_name, data2_name, 
                                        results, time_pd, duration, event_num)

            ### add these points to the full dataframe for 10-year
            exceed10_df_all = pd.concat([exceed10_df_all,df_event])

            #### change variable names here too - should be "prism_this", etc., if you're evaluating prism
            ### and deal with the removed points
            removed10_df_this = parse_latlons(prism_this,results['removed_data1_exceed10'],prism_exceed10_df,event_num,time_pd)
            removed10_df_all = pd.concat([removed10_df_all,removed10_df_this])

            ### for higher ARIs, find lat/lon points that match the exceedances
            if prism_exceed100_df is not None:
                exceed100_df_this = parse_latlons(prism_this,results['data1_exceed100'],prism_exceed100_df,event_num,time_pd)
                exceed100_df_all = pd.concat([exceed100_df_all,exceed100_df_this])
                removed100_df_this = parse_latlons(prism_this,results['removed_data1_exceed100'],prism_exceed100_df,event_num,time_pd)
                removed100_df_all = pd.concat([removed100_df_all,removed100_df_this])

            if prism_exceed1000_df is not None:
                exceed1000_df_this = parse_latlons(prism_this,results['data1_exceed1000'],prism_exceed1000_df,event_num,time_pd)
                exceed1000_df_all = pd.concat([exceed1000_df_all,exceed1000_df_this])
                removed1000_df_this = parse_latlons(prism_this,results['removed_data1_exceed1000'],prism_exceed1000_df,event_num,time_pd)
                removed1000_df_all = pd.concat([removed1000_df_all,removed1000_df_this])


### now write these to csv
exceed10_df_all.reset_index(drop=True).to_csv(dataset+"/auto_qc/"+str(duration).zfill(2)+"h/"+str(year)+"/"+dataset+"_"+str(year)+"_all_points_10y"+str(duration).zfill(2)+"h.csv", index=None)
exceed100_df_all.reset_index(drop=True).to_csv(dataset+"/auto_qc/"+str(duration).zfill(2)+"h/"+str(year)+"/"+dataset+"_"+str(year)+"_all_points_100y"+str(duration).zfill(2)+"h.csv", index=None)
exceed1000_df_all.reset_index(drop=True).to_csv(dataset+"/auto_qc/"+str(duration).zfill(2)+"h/"+str(year)+"/"+dataset+"_"+str(year)+"_all_points_1000y"+str(duration).zfill(2)+"h.csv", index=None)

removed10_df_all.reset_index(drop=True).to_csv(dataset+"/auto_qc/"+str(duration).zfill(2)+"h/"+str(year)+"/"+dataset+"_"+str(year)+"_removed_points_10y"+str(duration).zfill(2)+"h.csv", index=None)
removed100_df_all.reset_index(drop=True).to_csv(dataset+"/auto_qc/"+str(duration).zfill(2)+"h/"+str(year)+"/"+dataset+"_"+str(year)+"_removed_points_100y"+str(duration).zfill(2)+"h.csv", index=None)
removed1000_df_all.reset_index(drop=True).to_csv(dataset+"/auto_qc/"+str(duration).zfill(2)+"h/"+str(year)+"/"+dataset+"_"+str(year)+"_removed_points_1000y"+str(duration).zfill(2)+"h.csv", index=None)

### and get the cleaned dataframes (all points minus removed) and write them to csv too
cleaned10_df_all = get_cleaned_df(exceed10_df_all.reset_index(drop=True),removed10_df_all.reset_index(drop=True))
cleaned100_df_all = get_cleaned_df(exceed100_df_all.reset_index(drop=True),removed100_df_all.reset_index(drop=True))
cleaned1000_df_all = get_cleaned_df(exceed1000_df_all.reset_index(drop=True),removed1000_df_all.reset_index(drop=True))
cleaned10_df_all.reset_index(drop=True).to_csv(dataset+"/auto_qc/"+str(duration).zfill(2)+"h/"+str(year)+"/"+dataset+"_"+str(year)+"_cleaned_10y"+str(duration).zfill(2)+"h.csv", index=None)
cleaned100_df_all.reset_index(drop=True).to_csv(dataset+"/auto_qc/"+str(duration).zfill(2)+"h/"+str(year)+"/"+dataset+"_"+str(year)+"_cleaned_100y"+str(duration).zfill(2)+"h.csv", index=None)
cleaned1000_df_all.reset_index(drop=True).to_csv(dataset+"/auto_qc/"+str(duration).zfill(2)+"h/"+str(year)+"/"+dataset+"_"+str(year)+"_cleaned_1000y"+str(duration).zfill(2)+"h.csv", index=None)

print("done!")




