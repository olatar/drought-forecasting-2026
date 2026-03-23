import pandas as pd
import xarray as xr
import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import copy

def load_netcdf_as_xarray(filename, directory="data.nosync/saves.nosync"):
    file_path = os.path.join(directory, filename)

    dataset_xa = xr.open_dataset(file_path)
    print(f"Loaded dataset from {file_path}")

    return dataset_xa

def save_xarray_as_netcdf(dataset, filename, folder_path="data.nosync/saves.nosync"):
    """
    Save an xarray dataset as a NetCDF file

    Parameters:
    dataset (xarray.Dataset): The dataset to save
    folder_path (str): The path to the folder where the NetCDF file will be saved
    filename (str): The name of the NetCDF file (default is '/data/saves')
    """
    if filename.endswith(".nc"):
        filename = filename[:-3]

    today_yymmdd = datetime.datetime.now().strftime("%y%m%d")
    output_path = os.path.join(folder_path, filename + "_" + today_yymmdd + ".nc")

    dataset.to_netcdf(output_path)
    print(f"Saved dataset to {output_path}")



def rr_change_date_format_from_1D_to_2D(dataset: xr.Dataset, verbose=True):

    years = np.unique(dataset["time"].dt.year.values)
    months = np.unique(dataset["time"].dt.month.values)

    xarray = xr.DataArray(
        data=np.nan,  # Initialize with NaNs
        dims=["year", "month"],
        coords={
            'year': copy.deepcopy(years),
            'month': copy.deepcopy(months)
        }
    )

    dataset_year_month = xr.Dataset({"rr": xarray})
    #loop through each dimension in eobs_monthly and populate the eobs_year_month dataset
    times = dataset["time"].values

    for time in times:
        if verbose:
            print("Time {} of {}".format(time, times[-1]))
        
        year = pd.to_datetime(time).year
        month = pd.to_datetime(time).month

        rr_value = float(dataset.sel(time=time)['rr'])
        dataset_year_month['rr'].loc[dict(year=year, month=month)] = rr_value

    return dataset_year_month


def pp_change_date_format_from_1D_to_2D(dataset: xr.Dataset, verbose=True):

    years = np.unique(dataset["time"].dt.year.values)
    months = np.unique(dataset["time"].dt.month.values)

    xarray = xr.DataArray(
        data=np.nan,  # Initialize with NaNs
        dims=["year", "month"],
        coords={
            'year': copy.deepcopy(years),
            'month': copy.deepcopy(months)
        }
    )

    dataset_year_month = xr.Dataset({"pp": xarray})
    #loop through each dimension in eobs_monthly and populate the eobs_year_month dataset
    times = dataset["time"].values

    for time in times:
        if verbose:
            print("Time {} of {}".format(time, times[-1]))
        
        year = pd.to_datetime(time).year
        month = pd.to_datetime(time).month

        rr_value = float(dataset.sel(time=time)['pp'])
        dataset_year_month['pp'].loc[dict(year=year, month=month)] = rr_value

    return dataset_year_month


def tg_change_date_format_from_1D_to_2D(dataset: xr.Dataset, verbose=True):

    years = np.unique(dataset["time"].dt.year.values)
    months = np.unique(dataset["time"].dt.month.values)

    xarray = xr.DataArray(
        data=np.nan,  # Initialize with NaNs
        dims=["year", "month"],
        coords={
            'year': copy.deepcopy(years),
            'month': copy.deepcopy(months)
        }
    )

    dataset_year_month = xr.Dataset({"tg": xarray})
    #loop through each dimension in eobs_monthly and populate the eobs_year_month dataset
    times = dataset["time"].values

    for time in times:
        if verbose:
            print("Time {} of {}".format(time, times[-1]))
        
        year = pd.to_datetime(time).year
        month = pd.to_datetime(time).month

        rr_value = float(dataset.sel(time=time)['tg'])
        dataset_year_month['tg'].loc[dict(year=year, month=month)] = rr_value

    return dataset_year_month

def rainyrate_change_date_format_from_1D_to_2D(dataset: xr.Dataset, verbose=True):

    years = np.unique(dataset["time"].dt.year.values)
    months = np.unique(dataset["time"].dt.month.values)

    xarray = xr.DataArray(
        data=np.nan,  # Initialize with NaNs
        dims=["year", "month"],
        coords={
            'year': copy.deepcopy(years),
            'month': copy.deepcopy(months)
        }
    )

    dataset_year_month = xr.Dataset({"rainy_days_rate": xarray})
    #loop through each dimension in eobs_monthly and populate the eobs_year_month dataset
    times = dataset["time"].values

    for time in times:
        if verbose:
            print("Time {} of {}".format(time, times[-1]))
        
        year = pd.to_datetime(time).year
        month = pd.to_datetime(time).month

        rr_value = float(dataset.sel(time=time)['rainy_days_rate'])
        dataset_year_month['rainy_days_rate'].loc[dict(year=year, month=month)] = rr_value

    return dataset_year_month

def print_characteristics(dataset: xr.Dataset, element, ecmwf=False):
    if not ecmwf:
        print('Time dimension size: {}'.format(dataset['time'].shape))
        print('Time granularity: 1 day')
        print('Time period: from {} to {}'.format(pd.to_datetime(dataset['time'][0].values).date(), pd.to_datetime(dataset['time'][-1].values).date()))

    print('\nLongitude dimension size: {}'.format(dataset['longitude'].shape))
    print('Latitude dimension size: {}'.format(dataset['latitude'].shape))

    if ecmwf:
        print('\nPrecipitation dimensions: {}'.format(dataset['tprate'].dims))
        print('Precipitation dimension size: {}'.format(dataset['tprate'].shape))
    else:
        print('\nPrecipitation dimensions: {}'.format(dataset[element].dims))
        print('Precipitation dimension size: {}'.format(dataset[element].shape))

    print('Precipitation unit: daily precipitation sum')

    min_lat = dataset.latitude.min().values
    max_lat = dataset.latitude.max().values
    min_lon = dataset.longitude.min().values
    max_lon = dataset.longitude.max().values

    print('')
    print('Latitude min/max: {} / {}'.format(min_lat, max_lat))
    print('Longitude min/max: {} / {}'.format(min_lon, max_lon))


def aggregate_to_monthly(eobs_dataset: xr.Dataset):
    monthly = eobs_dataset.resample(time='1M').sum()
    return monthly


def select_subregion_from_single_coordinate(dataset: xr.Dataset, lat, lon):
    print('Selecting subregion around lat: {:.4f} and lon: {:.4f}'.format(lat, lon))
    print('Closest coordinates: lat: {:.4f} and lon: {:.4f}'.format(
        dataset.sel(latitude=lat, method='nearest')['latitude'].values,
        dataset.sel(longitude=lon, method='nearest')['longitude'].values))

    return dataset.sel(latitude=lat, longitude=lon, method='nearest')


def print_temporal_information(dataset: xr.Dataset):
    print('Time dimension size: {}'.format(dataset['time'].shape))
    
    time_diff = dataset['time'][1] - dataset['time'][0]
    print('Time granularity: {}'.format(time_diff.values))


    print('Time period: from {} to {}'.format(pd.to_datetime(dataset['time'][0].values).date(),
                                              pd.to_datetime(dataset['time'][-1].values).date()))


def sum_across_lat_lon(dataset):
    return dataset.sum(dim=["longitude", "latitude"])


def mean_across_lat_lon(dataset):
    return dataset.mean(dim=["longitude", "latitude"])


def select_from_start_year(dataset: xr.Dataset, year_start, year_end):
    return dataset.sel(year=slice(year_start, year_end))


def plot_map_with_locations(dataset: xr.Dataset):

    # Extract latitude and longitude
    latitudes = dataset['latitude'].values
    longitudes = dataset['longitude'].values

    # Create a meshgrid for plotting
    lon_grid, lat_grid = np.meshgrid(longitudes, latitudes)

    # Create a plot with a map underneath
    plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([longitudes.min(), longitudes.max(), latitudes.min(), latitudes.max()])

    # Add natural Earth features
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')

    # Plot the latitude and longitude grid
    ax.scatter(lon_grid, lat_grid, color='red', s=10, transform=ccrs.PlateCarree())

    # Add gridlines for better orientation
    ax.gridlines(draw_labels=True, crs=ccrs.PlateCarree())

    plt.title('Latitude and Longitude Grid with Map')
    plt.show()