### some functions suggested by claude for comparing datasets for QC

import pandas as pd
import xarray as xr
import os
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.offsetbox import AnchoredText

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import (LongitudeFormatter, LatitudeFormatter,
                                LatitudeLocator)

from scipy.ndimage import label, gaussian_filter
from scipy.stats import pearsonr
from sklearn.cluster import DBSCAN

from ari_exceedance_map_functions import *

def compare_datasets_difference(data1, data2, difference_threshold=10.0, 
                               ratio_threshold=4.0, min_cluster_size=12):
    """
    Identify artifacts by finding large differences between datasets
    """
    # Calculate absolute and relative differences
    abs_diff = np.abs(data1 - data2)
    
    # Avoid division by zero
    denominator = np.maximum(data2, 0.1)  # minimum threshold
    ratio_diff = data1 / denominator
    
    # Identify potential artifacts in dataset 1
    artifacts_1 = (abs_diff > difference_threshold) & (ratio_diff > ratio_threshold)
    
    # Identify potential artifacts in dataset 2  
    ratio_diff_2 = data2 / np.maximum(data1, 0.1)
    artifacts_2 = (abs_diff > difference_threshold) & (ratio_diff_2 > ratio_threshold)
    
    # Remove small isolated differences (likely noise)
    artifacts_1 = remove_small_clusters(artifacts_1, min_cluster_size)
    artifacts_2 = remove_small_clusters(artifacts_2, min_cluster_size)
    
    return artifacts_1, artifacts_2, abs_diff

def remove_small_clusters(binary_mask, min_size):
    """Remove clusters smaller than min_size"""
    labeled, num_features = label(binary_mask)
    for i in range(1, num_features + 1):
        cluster = labeled == i
        if np.sum(cluster) < min_size:
            binary_mask[cluster] = True
    return binary_mask

def correlation_based_detection(data1, data2, window_size=5, correlation_threshold=0.25):
    """
    Identify areas where local correlation between datasets is poor
    """
    rows, cols = data1.shape
    correlation_map = np.full((rows, cols), np.nan)
    
    half_window = window_size // 2
    
    for i in range(half_window, rows - half_window):
        for j in range(half_window, cols - half_window):
            # Extract local windows
            window1 = data1[i-half_window:i+half_window+1, 
                           j-half_window:j+half_window+1].flatten()
            window2 = data2[i-half_window:i+half_window+1, 
                           j-half_window:j+half_window+1].flatten()
            
            # Remove NaN values
            valid_mask = ~(np.isnan(window1) | np.isnan(window2))
            if np.sum(valid_mask) > 5:  # Need enough valid points
                corr, _ = pearsonr(window1[valid_mask], window2[valid_mask])
                correlation_map[i, j] = corr
    
    # Identify areas with poor correlation
    poor_correlation = correlation_map < correlation_threshold
    
    return correlation_map, poor_correlation

def residual_analysis(data1, data2, smooth_sigma=2, residual_threshold=3):
    """
    Compare datasets after smoothing to identify sharp, localized differences
    """
    # Smooth both datasets
    smooth1 = gaussian_filter(data1, sigma=smooth_sigma)
    smooth2 = gaussian_filter(data2, sigma=smooth_sigma)
    
    # Calculate residuals (difference from smooth version)
    residual1 = data1 - smooth1
    residual2 = data2 - smooth2
    
    # Calculate difference in residuals
    residual_diff = np.abs(residual1 - residual2)
    
    # Identify areas where residual differences are large
    threshold = np.nanpercentile(residual_diff, 95) + residual_threshold * np.nanstd(residual_diff)
    artifacts = residual_diff > threshold
    
    return residual1, residual2, residual_diff, artifacts

def dual_dataset_outlier_detection(data1, data2, z_threshold=3):
    """
    Use both datasets to identify statistical outliers
    """
    # Stack both datasets
    combined = np.stack([data1, data2], axis=0)
    
    # Calculate statistics across both datasets
    mean_combined = np.nanmean(combined, axis=0)
    std_combined = np.nanstd(combined, axis=0)
    
    # Calculate z-scores for each dataset
    z_score1 = np.abs(data1 - mean_combined) / (std_combined + 1e-8)
    z_score2 = np.abs(data2 - mean_combined) / (std_combined + 1e-8)
    
    # Identify outliers
    outliers1 = z_score1 > z_threshold
    outliers2 = z_score2 > z_threshold
    
    return outliers1, outliers2, z_score1, z_score2

def comprehensive_dataset_comparison(data1, data2, 
                                     data1_exceed10, data2_exceed10,
                                     data1_exceed100, data2_exceed100,
                                     data1_exceed1000, data2_exceed1000,
                                     data1_name="Dataset 1", data2_name="Dataset 2"):
    """
    Apply multiple comparison methods and return comprehensive results
    """
    results = {}
    
    # Method 1: Difference-based detection
    artifacts_1_diff, artifacts_2_diff, abs_diff = compare_datasets_difference(data1, data2)
    results['difference_artifacts_1'] = artifacts_1_diff
    results['difference_artifacts_2'] = artifacts_2_diff
    results['absolute_difference'] = abs_diff
    
    # Method 2: Correlation analysis (not being used)
    #corr_map, poor_corr = correlation_based_detection(data1, data2)
    #results['correlation_map'] = corr_map
    #results['poor_correlation_areas'] = poor_corr
    
    # Method 3: Residual analysis
    res1, res2, res_diff, res_artifacts = residual_analysis(data1, data2)
    results['residual_artifacts'] = res_artifacts
    results['residual_difference'] = res_diff
    
    # Method 4: Outlier detection
    outliers1, outliers2, z1, z2 = dual_dataset_outlier_detection(data1, data2)
    results['outliers_1'] = outliers1
    results['outliers_2'] = outliers2
    
    # Combined suspicious areas
    suspicious_1 = artifacts_1_diff | outliers1
    suspicious_2 = artifacts_2_diff | outliers2
    
    results['combined_suspicious_1'] = suspicious_1
    results['combined_suspicious_2'] = suspicious_2

    ### put all the exceedance points in the results
    results['data1_exceed10'] = data1_exceed10
    results['data1_exceed100'] = data1_exceed100
    results['data1_exceed1000'] = data1_exceed1000

    ### and find exceedance points that have been removed
    combined_artifacts = results['combined_suspicious_1'] | results['combined_suspicious_2']
    results['removed_data1_exceed10'] = data1_exceed10 & combined_artifacts
    results['removed_data2_exceed10'] = data2_exceed10 & combined_artifacts
    results['removed_data1_exceed100'] = data1_exceed100 & combined_artifacts
    results['removed_data2_exceed100'] = data2_exceed100 & combined_artifacts
    results['removed_data1_exceed1000'] = data1_exceed1000 & combined_artifacts
    results['removed_data2_exceed1000'] = data2_exceed1000 & combined_artifacts

    # Summary statistics
    print('')
    print(f"Potential artifacts in {data1_name}: {np.sum(suspicious_1)} pixels")
    print(f"Potential artifacts in {data2_name}: {np.sum(suspicious_2)} pixels")
    print(f"Overall correlation: {pearsonr(data1.flatten(), data2.flatten())[0]:.3f}")
    print('')

    ### number of exceedances before and after
    print(str(np.count_nonzero(results['removed_data1_exceed10']))+" of "+str(np.count_nonzero(data1_exceed10))+" 10-y exceedances in "+data1_name+" removed by cleaning")
    print(str(np.count_nonzero(results['removed_data1_exceed100']))+" of "+str(np.count_nonzero(data1_exceed100))+" 100-y exceedances in "+data1_name+" removed by cleaning")
    print(str(np.count_nonzero(results['removed_data1_exceed1000']))+" of "+str(np.count_nonzero(data1_exceed1000))+" 1000-y exceedances in "+data1_name+" removed by cleaning")
    print('')
    #print("10-y exceedances in "+data2_name+": "+str(np.count_nonzero(data2_exceed10)))
    #print("10-y exceedances in "+data2_name+" removed by cleaning: "+str(np.count_nonzero(results['removed_data2_exceed10'])))
        
    return results

def plot_comparison_results(data1, data2, 
                            data1_exceed10, data2_exceed10, 
                            data1_exceed100, data2_exceed100, 
                            data1_exceed1000, data2_exceed1000, 
                            lons, lats, this_lon, this_lat,
                            minlon, maxlon, minlat, maxlat, cmap, norm,
                            dataset, data1_name, data2_name, results, 
                            time_pd, duration, event_num,
                            figsize=(11.5, 8.)):
    
    """
    Create comprehensive visualization of comparison results
    """
    crs = ccrs.LambertConformal(central_longitude=this_lon, central_latitude=this_lat)

    fig, axes = plt.subplots(3, 4, figsize=figsize,
                            layout='constrained',
                            subplot_kw={'projection': crs})

    axlist = axes.flatten()
    for ax in axlist:
        plot_background(ax,minlon,maxlon,minlat,maxlat)

    lon2d, lat2d = np.meshgrid(lons, lats)
    
    # Original datasets
    #im1 = axes[0,0].imshow(data1, cmap='viridis')
    im1 = axes[0,0].pcolormesh(lon2d, lat2d, data1, cmap=cmap, norm=norm,
                          transform=ccrs.PlateCarree())
    axes[0,0].set_title(data1_name, fontsize=10)
    #plt.colorbar(im1, ax=axes[0,0], pad=0)
    
    im2 = axes[1,0].pcolormesh(lon2d, lat2d, data2, cmap=cmap, norm=norm,
                          transform=ccrs.PlateCarree())
    axes[1,0].set_title(data2_name, fontsize=10)
    #plt.colorbar(im2, ax=axes[1,0], pad=0)

    # Define colors: transparent for False, red for True
    colors10 = [(0, 0, 0, 0), (1, 0.5, 0.5, 0.85)]  # RGBA: (R, G, B, Alpha)
    cmap_points10 = mcolors.ListedColormap(colors10)

    colors100 = [(0, 0, 0, 0), (1, 0.25, 0.25, 0.925)]  # RGBA: (R, G, B, Alpha)
    cmap_points100 = mcolors.ListedColormap(colors100)

    colors1000 = [(0, 0, 0, 0), (0.9, 0.1, 0.1, 1)]  # RGBA: (R, G, B, Alpha)
    cmap_points1000 = mcolors.ListedColormap(colors1000)
   
    ### exceedance points
    im1_10 = axes[0,1].pcolormesh(lon2d, lat2d, data1_exceed10,cmap=cmap_points10,transform=ccrs.PlateCarree())
    im1_100 = axes[0,1].pcolormesh(lon2d, lat2d, data1_exceed100,cmap=cmap_points100,transform=ccrs.PlateCarree())
    im1_1000 = axes[0,1].pcolormesh(lon2d, lat2d, data1_exceed1000,cmap=cmap_points1000,transform=ccrs.PlateCarree())
    axes[0,1].annotate(str(np.count_nonzero(data1_exceed10))+" 10y points\n"+str(np.count_nonzero(data1_exceed100))+" 100y points\n"+str(np.count_nonzero(data1_exceed1000))+" 1000y points",
                       xy=(0, 0.07), xycoords='axes fraction',
                xytext=(2, 20), textcoords='offset points', color='red',
                ha='left', va='top', fontsize=8)
    axes[0,1].set_title(data1_name+" exceedances", fontsize=10)

    ### exceedance points
    im1_10 = axes[1,1].pcolormesh(lon2d, lat2d, data2_exceed10,cmap=cmap_points10,transform=ccrs.PlateCarree())
    im1_100 = axes[1,1].pcolormesh(lon2d, lat2d, data2_exceed100,cmap=cmap_points100,transform=ccrs.PlateCarree())
    im1_1000 = axes[1,1].pcolormesh(lon2d, lat2d, data2_exceed1000,cmap=cmap_points1000,transform=ccrs.PlateCarree())
    axes[1,1].annotate(str(np.count_nonzero(data2_exceed10))+" 10y points\n"+str(np.count_nonzero(data2_exceed100))+" 100y points\n"+str(np.count_nonzero(data2_exceed1000))+" 1000y points",
                       xy=(0, 0.07), xycoords='axes fraction',
                xytext=(2, 20), textcoords='offset points', color='red',
                ha='left', va='top', fontsize=8)
    axes[1,1].set_title(data2_name+" exceedances", fontsize=10)
    
    # Absolute difference
    im3 = axes[2,0].pcolormesh(lon2d, lat2d, results['absolute_difference'], cmap='Reds',
                          transform=ccrs.PlateCarree())
    axes[2,0].set_title('Absolute Difference', fontsize=10)
    #plt.colorbar(im3, ax=axes[2,0],pad=0)
    
    # Correlation map
    im4 = axes[2,1].pcolormesh(lon2d, lat2d, results['correlation_map'], cmap='RdYlBu', vmin=-1, vmax=1,
                          transform=ccrs.PlateCarree())
    axes[2,1].set_title('Local Correlation', fontsize=10)
    #plt.colorbar(im4, ax=axes[2,1],pad=0)
    
    # Suspected artifacts
    axes[0,2].pcolormesh(lon2d, lat2d, results['combined_suspicious_1'], cmap='Reds', alpha=0.7,
                    transform=ccrs.PlateCarree())
    axes[0,2].pcolormesh(lon2d, lat2d, data1, cmap=cmap, norm=norm, alpha=0.3, 
                         transform=ccrs.PlateCarree())
    axes[0,2].set_title(data1_name+' - Suspected Artifacts', fontsize=10)

    # Residual analysis
    im7 = axes[0,3].pcolormesh(lon2d, lat2d, results['residual_difference'], cmap='Reds',
                          transform=ccrs.PlateCarree())
    axes[0,3].set_title('Residual Differences', fontsize=10)
    #plt.colorbar(im7, ax=axes[0,3],pad=0)
    
    axes[1,2].pcolormesh(lon2d, lat2d, results['combined_suspicious_2'], cmap='Reds', alpha=0.7,
                    transform=ccrs.PlateCarree())
    axes[1,2].pcolormesh(lon2d, lat2d, data2, cmap=cmap, norm=norm, alpha=0.3, 
                         transform=ccrs.PlateCarree())
    axes[1,2].set_title(data2_name +' - Suspected Artifacts', fontsize=10)
    
    # Poor correlation areas
    axes[1,3].pcolormesh(lon2d, lat2d, results['poor_correlation_areas'], cmap='Reds',
                    transform=ccrs.PlateCarree())
    axes[1,3].set_title('Poor Correlation Areas', fontsize=10)
    
    # Combined view
    combined_artifacts = results['combined_suspicious_1'] | results['combined_suspicious_2']
    axes[2,2].pcolormesh(lon2d, lat2d, combined_artifacts, cmap='Reds', alpha=0.7, transform=ccrs.PlateCarree())
    axes[2,2].pcolormesh(lon2d, lat2d, (data1 + data2)/2, cmap=cmap, norm=norm, alpha=0.3, 
                         transform=ccrs.PlateCarree())
    axes[2,2].set_title('All Suspected Artifacts', fontsize=10)

    axes[2,3].pcolormesh(lon2d, lat2d, results['removed_data1_exceed10'], cmap=cmap_points10, transform=ccrs.PlateCarree())
    axes[2,3].pcolormesh(lon2d, lat2d, results['removed_data1_exceed100'], cmap=cmap_points100, transform=ccrs.PlateCarree())
    axes[2,3].pcolormesh(lon2d, lat2d, results['removed_data1_exceed1000'], cmap=cmap_points1000, transform=ccrs.PlateCarree())
    axes[2,3].annotate(str(np.count_nonzero(results['removed_data1_exceed10']))+" of "+str(np.count_nonzero(data1_exceed10))+" "+data1_name+" 10y points removed\n"+str(np.count_nonzero(results['removed_data1_exceed100']))+" of "+str(np.count_nonzero(data1_exceed100))+" "+data1_name+" 100y points removed\n"+str(np.count_nonzero(results['removed_data1_exceed1000']))+" of "+str(np.count_nonzero(data1_exceed1000))+" "+data1_name+" 1000y points removed",
                       xy=(0, 0.07), xycoords='axes fraction', color='red',
                xytext=(2, 20), textcoords='offset points',
                ha='left', va='top', fontsize=8.)
    axes[2,3].set_title('Removed exceedance points', fontsize=10)

    plt.suptitle(str(duration)+"-hr exceedances, valid "+time_pd.strftime("%H%M UTC %d %b %Y"))
    os.system("mkdir -p "+dataset+"/auto_qc/"+str(duration).zfill(2)+"h/"+time_pd.strftime("%Y")+"/maps")
    plt.savefig(dataset+"/auto_qc/"+str(duration).zfill(2)+"h/"+time_pd.strftime("%Y")+"/maps/qc_check_"+time_pd.strftime("%Y%m%d%H")+"_"+str(event_num).zfill(1)+".png",
                dpi=175,facecolor='white',transparent=False,bbox_inches='tight')
    #plt.show()
    plt.close('all')


### this function does distance-based clustering to find all of the individual events on a day
def cluster_points_dbscan(df, max_distance_km=50, min_points=3):
    """
    Use DBSCAN to find natural clusters of points
    """
    # Convert to approximate Cartesian coordinates (for small regions)
    # More accurate for larger areas: use projected coordinates
    coords = df[['lat', 'lon']].values
    
    # For geographic coordinates, we need to convert distance to degrees approximately
    # 1 degree ≈ 111 km at equator (varies by latitude)
    max_distance_deg = max_distance_km / 111.0
    
    # Apply DBSCAN
    clustering = DBSCAN(eps=max_distance_deg, min_samples=min_points).fit(coords)
    
    # Add cluster labels to dataframe
    df_clustered = df.copy()
    df_clustered['event_num'] = clustering.labels_
    
    # Analyze clusters
    #cluster_summary = df_clustered.groupby('event_num').agg({
    #    'lat': ['count', 'mean', 'std'],
    #    'lon': ['mean', 'std']
    #}).round(4)
    
    n_clusters = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
    #n_noise = list(clustering.labels_).count(-1)
    
    #print(f"Number of clusters: {n_clusters}")
    #print(f"Number of noise points: {n_noise}")
    
    return df_clustered, n_clusters

### function to create a pandas dataframe from an xarray dataset containing points exceeding an ARI
### inputs are the xarray dataset, the relevant ARI grid, and the current datetime
### output is a dataframe
def create_exceed_df(ds,ari_grid,time_pd):
    #### calculate how much above the ARI threshold each point is
    ds['tp_minus_ari'] = ds.tp - ari_grid.precip
    ds['tp_pct_of_ari'] = ds.tp/ari_grid.precip
    ### put these in pandas dataframes
    tp_df = ds.tp.to_dataframe().reset_index().dropna()
    tp_minus_ari_df = ds.tp_minus_ari.to_dataframe().reset_index().dropna()
    pct_ari_df = ds.tp_pct_of_ari.to_dataframe().reset_index().dropna()
    
    exceed_df = tp_df.merge(tp_minus_ari_df)
    exceed_df = exceed_df.merge(pct_ari_df)
    exceed_df['time'] = time_pd  ### add a time column

    return exceed_df
    

### function to parse lat/lons from the indexes returned by numpy,
#### and match them with the lat/lon points from an exceedance dataframe
### inputs are an xarray dataarray (like "mrms_this"), the "exceed" variable from the QC results, 
#### and the big exceed dataframe for the desired threshold. and event_num and time_pd
### output is the dataframe, now with the event/cluster number included

def parse_latlons(grid_da,results_exceed,exceed_df,event_num,time_pd):
    all_lons = grid_da.isel(lon=results_exceed.nonzero()[1]).lon.values
    all_lats = grid_da.isel(lat=results_exceed.nonzero()[0]).lat.values

    ### put into a dataframe
    all_latlons = pd.DataFrame([all_lons,all_lats]).T
    all_latlons.columns=(['lon','lat'])

    merged = pd.merge(all_latlons,exceed_df,on=['lon','lat'],how='inner')
    exceed_df_this = merged[merged.time==time_pd.strftime("%Y-%m-%d %H:%M:%S")]
    exceed_df_this['event_num'] = event_num

    cols=['time','lat','lon','tp','tp_minus_ari','tp_pct_of_ari','event_num']
    exceed_df_this = exceed_df_this[cols]
    
    return exceed_df_this

#### function to get the cleaned dataframe after removing the removed ones...
def get_cleaned_df(exceed_df,removed_df):
    merged_df = pd.merge(exceed_df, removed_df, 
                            on=['time','lon','lat','tp','event_num','tp_minus_ari','tp_pct_of_ari'], 
                            how='left', indicator=True)
    # Filter out the rows present in both Data Frames
    cleaned_df = merged_df[merged_df['_merge'] == 'left_only']
    # Remove the indicator column
    cleaned_df = cleaned_df.drop(columns='_merge')

    return cleaned_df

