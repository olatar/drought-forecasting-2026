import os
import xarray as xr
from utils import load_netcdf_as_xarray, save_xarray_as_netcdf
import datetime
import calendar
from scipy import stats
import numpy as np
import copy

MONTHS = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]


def load_ecmwf_netcdfs_to_xarray(directory):

    nc_files = [
        os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".nc")
    ]
    nc_files_sorted = sorted(
        nc_files, key=lambda x: int(x.split("_")[-1].split(".")[0][2:])
    )

    datasets_dict = {}
    for file in nc_files_sorted:
        # Assuming that 'imX' in filename corresponds to month number X
        month_num = int(
            file.split("_")[-1].split(".")[0][2:]
        )  # Extract the month number
        month_name = MONTHS[
            month_num - 1
        ]  # Get the corresponding month name from the list
        dataset = xr.open_dataset(file)
        datasets_dict[month_name] = (
            dataset  # Use the month name as the key in the dictionary
        )
        print(f"Loaded file for {month_name}: {file}")

    datasets_list = [datasets_dict[month] for month in MONTHS]

    # Rename 'month' variable if it exists to avoid conflict
    for i, (month, dataset) in enumerate(zip(MONTHS, datasets_list)):
        dataset = dataset.expand_dims("month").assign_coords(month=[month])
        datasets_list[i] = dataset

    dataset_xa = xr.concat(datasets_list, dim="month")

    # -------
    january_subset = dataset_xa.sel(month="March").drop_vars("month")

    are_identical = datasets_dict["March"].equals(january_subset)
    print(f"Are the datasets identical? {are_identical}")
    # -------

    return dataset_xa


def save_xarray_as_netcdf(dataset, filename, folder_path="data/saves"):
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


def load_netcdf_as_xarray(filename, most_recent=True, directory="data/saves"):
    """
    Load a NetCDF file as an xarray dataset

    Parameters:
    file_path (str): The path to the NetCDF file

    Returns:
    xarray.Dataset: The loaded dataset
    """
    if filename.endswith(".nc"):
        file_path = os.path.join(directory, filename)
    else:
        if most_recent:
            latest_date = datetime.datetime.min
            latest_filename = None
            for file in os.listdir(directory):
                if file.endswith(".nc"):
                    filename_temp = file.split(".")[0][:-7]
                    date = file.split(".")[0].split("_")[-1]
                    date_formatted = datetime.datetime.strptime(date, "%y%m%d")

                    if date_formatted > latest_date and filename == filename_temp:
                        latest_date = date_formatted
                        latest_filename = file

            if latest_filename is not None:
                file_path = os.path.join(directory, latest_filename)
            else:
                raise FileNotFoundError("No NetCDF files found in the specified directory")
        else:
            file_path = os.path.join(directory, filename + ".nc")

    dataset_xa = xr.open_dataset(file_path)
    print(f"Loaded dataset from {file_path}")

    return dataset_xa


def rename_netcdf_dimension(dataset, dict, save=False, save_filename=None):
    dataset = dataset.rename(dict)
    if save:
        save_xarray_as_netcdf(dataset, save_filename)
    return dataset


def find_missing_years(datasets_dict, possible_years=range(1993, 2024)):

    months = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]
    for month_name in months:

        available_years = set(datasets_dict[month_name].year.values)
        missing_years = possible_years - available_years
        print(f"Missing years for {month_name}: {sorted(missing_years)}")


def remove_year(dataset, year, save=False, save_filename=None):
    dataset = dataset.sel(year=~dataset.year.isin([year]))
    if save:
        save_xarray_as_netcdf(dataset, save_filename)
    return dataset


# dataset_xa.sel(ensemble_member=slice(0, 24))
def select_ensemble_members(dataset, start=0, end=24, save=False, save_filename=None):
    dataset = dataset.sel(ensemble_member=slice(start, end))
    if save:
        save_xarray_as_netcdf(dataset, save_filename)
    return dataset


# get number of days given a year and month
def days_per_year_month(year: int, month: str):
    month: int = list(calendar.month_abbr).index(month[:3].capitalize())
    return calendar.monthrange(year, month)[1]


def month_idx_as_string(reference_month, month_number):

    reference_month_index = list(calendar.month_abbr).index(
        reference_month[:3].capitalize()
    )

    target_month_index = (reference_month_index + month_number - 2) % 12

    month_string = calendar.month_name[target_month_index + 1]
    return month_string


def get_number_of_days(reference_year, reference_month, time_index):
    # Get the month string based on the reference month and time index
    forecast_month = month_idx_as_string(reference_month, time_index)

    if time_index > 1 and forecast_month == month_idx_as_string(reference_month, 1):
        next_year = reference_year + 1
    else:
        next_year = reference_year

    number_of_days = days_per_year_month(next_year, forecast_month)

    return number_of_days


def modify_per_second_to_per_month(
    dataset_xa, variable="tprate", seconds_in_day=86400, save=False, save_filename=None
):
    for year in dataset_xa.year.values:
        for month in dataset_xa.month.values:
            for forecast_index in dataset_xa.forecast_index.values:

                # Calculate the number of days for the respective month and y
                number_of_days = get_number_of_days(year, month, forecast_index)
                conversion_factor = seconds_in_day * number_of_days

                data_subset = dataset_xa.sel(
                    year=year, month=month, forecast_index=forecast_index
                )

                # Multiply the data by the conversion factor
                data_subset[variable] *= conversion_factor

                dataset_xa.loc[
                    {"year": year, "month": month, "forecast_index": forecast_index}
                ] = data_subset[variable]

    if save:
        save_xarray_as_netcdf(dataset_xa, save_filename)

    return dataset_xa


def sum_across_lat_lon(dataset_xa, save=False, save_filename=None):
    dataset_xa = dataset_xa.sum(dim=["longitude", "latitude"])
    if save:
        save_xarray_as_netcdf(dataset_xa, save_filename)
    return dataset_xa


def aggregate_across_ensemble_members(
    dataset_xa, method, save=False, save_filename=None
):
    if method == "mean":
        dataset_xa = dataset_xa.mean(dim=["ensemble_member"])
    elif method == "median":
        dataset_xa = dataset_xa.median(dim=["ensemble_member"])
    else:
        raise ValueError(
            "Invalid aggregation method. Choose between 'mean' and 'median'."
        )

    if save:
        save_xarray_as_netcdf(dataset_xa, save_filename)
    return dataset_xa


# def spi_of_timeseries(precip_data_each_year):

#     # Fit gamma distribution to positive values (floc forces loc to 0, thus output loc will be 0)
#     shape, loc, scale = stats.gamma.fit(precip_data_each_year, floc=0)

#     # GAMMA FIT & CDF VALUE: gamma_cdf = stats.gamma.cdf(adjusted_precip_data, a=shape, loc=loc, scale=scale)
#     gamma_cdf = stats.gamma.cdf(precip_data_each_year, a=shape, scale=scale, loc=loc)

#     # NORMAL DIst: Convert the gamma CDF to SPI values
#     spi_values = stats.norm.ppf(gamma_cdf)

#     return spi_values, gamma_cdf, shape, loc, scale


#######################################################################################################################
################################################## SPI-1 CALCULATION ##################################################
#######################################################################################################################

def create_spi_data_structure(dataset_xa, with_ensamble_member_dim=False, save=False, save_filename=None):
    # Create a new dataset with the desired dimensions and coordinates

    if with_ensamble_member_dim:
        xarray = xr.DataArray(
            data=np.nan,  # Initialize with NaNs
            dims=["year", "month", "forecast_index", "ensemble_member"],
            coords={
                "year": copy.deepcopy(dataset_xa.year.values),
                "month": copy.deepcopy(dataset_xa.month.values),
                "forecast_index": copy.deepcopy(dataset_xa.forecast_index.values),
                "ensemble_member": copy.deepcopy(dataset_xa.ensemble_member.values),
            },
            # name="SPI-1",
        )
    else:
        xarray = xr.DataArray(
            data=np.nan,  # Initialize with NaNs
            dims=["year", "month", "forecast_index"],
            coords={
                "year": copy.deepcopy(dataset_xa.year.values),
                "month": copy.deepcopy(dataset_xa.month.values),
                "forecast_index": copy.deepcopy(dataset_xa.forecast_index.values),
            },
            # name="SPI-1",
        )

    spi_1_dataset_xa = xr.Dataset({"spi_1_values": xarray})

    if save:
        save_xarray_as_netcdf(spi_1_dataset_xa, save_filename)

    return spi_1_dataset_xa


def calc_spi_1_values(dataset_xa):
    # from xarray datastructure, print all dimesions, using xarray dimensions function
    # print("Original dimensions xarray: ", list(dataset_xa.dims))
    
    dataset_np = np.array(dataset_xa['tprate'])

    dimensions = dataset_np.shape
    # print("Original dimensions: ", dimensions)
    dataset_np_flatten = dataset_np.flatten()
    
    # Fit gamma distribution to positive values (floc forces loc to 0, thus output loc will be 0)
    shape, loc, scale = stats.gamma.fit(dataset_np_flatten, floc=0)

    # GAMMA FIT & CDF VALUE: gamma_cdf = stats.gamma.cdf(adjusted_precip_data, a=shape, loc=loc, scale=scale)
    gamma_cdf = stats.gamma.cdf(dataset_np_flatten, a=shape, scale=scale, loc=loc)

    # NORMAL DIst: Convert the gamma CDF to SPI values
    spi_values = stats.norm.ppf(gamma_cdf)

    spi_values_original_dim = spi_values.reshape(dimensions)

    return spi_values_original_dim


def spi_1_from_sum_lat_lon(dataset_xa, save=False, save_filename=None):

    spi_datastructure = create_spi_data_structure(dataset_xa, with_ensamble_member_dim=True, save=False)

    all_month = dataset_xa.month.values
    all_forecast_index = dataset_xa.forecast_index.values
    all_year = dataset_xa.year.values
    all_ensemble_member = dataset_xa.ensemble_member.values

    for month in all_month:
        print("\nXXXXX month: {} nXXXXXn".format(month))

        for forecast_index in all_forecast_index:
            print("----- forecast_index: {} -----\n".format(forecast_index))
            data_distribution = dataset_xa.sel(month=month, forecast_index=forecast_index)

            # Calculate SPI values for the current month and forecast index (across all years and ensemble members)
            spi_values_na = calc_spi_1_values(data_distribution)

            spi_values_xa = xr.DataArray(spi_values_na, dims=['year', 'ensemble_member'], 
                                        coords={'year': all_year, 'ensemble_member': all_ensemble_member})

            # Loop through month, forecast_index, year, and ensemble_member to assign the SPI values to the data structure in an organized way
            for year in all_year:
                for ensemble_member in all_ensemble_member:

                    spi_value = spi_values_xa.sel(year=year, ensemble_member=ensemble_member).values
                    print("month: {}, forecast_idx: {}, year: {}, ensemble_member: {}, spi_value: {}".format(month, forecast_index, year, ensemble_member, spi_value))

                    spi_datastructure['spi_1_values'].loc[year, month, forecast_index, ensemble_member] = float(spi_value)

    if save:
        save_xarray_as_netcdf(spi_datastructure, save_filename)

    return spi_datastructure


#######################################################################################################################
############################################### LEAD TIME TRANSFORMATION ##############################################
#######################################################################################################################

def get_next_month(current_month, spi_index):
    current_month_index = list(calendar.month_abbr).index(current_month[:3].capitalize())
    next_month_index = current_month_index + spi_index - 1
    next_month_year = 1 if next_month_index > 12 else 0
    next_month_index %= 12
    if next_month_index == 0:
        next_month_index = 12
    next_month = calendar.month_name[next_month_index]
    return next_month, next_month_year


# Define a corrected function to determine if the forecasted month is in the same year or the next year
def get_forecast_year(year, month, spi_index):
    _, next_month_in_next_year = get_next_month(month, spi_index)
    if next_month_in_next_year:
        return year + 1
    else:
        return year


def test_get_next_month():
    # Test scenarios for get_next_month function
    test_cases_next_month = [
        ("January", 1, "January"),
        ("January", 2, "February"),
        ("November", 5, "March"),
        ("December", 2, "January"),
    ]

    # Test get_next_month function
    for current_month, spi_index, expected_next_month in test_cases_next_month:
        actual_next_month = get_next_month(current_month, spi_index)
        print(f"For {current_month} with spi_index {spi_index}, expected next month is {expected_next_month}. Actual: {actual_next_month}")

    print("")


def test_get_forecast_year():
    # Define corrected test scenarios for get_forecast_year function
    forecast_year_tests = [
        # (year, month, spi_index, expected_forecast_year)
        (1993, "December", 1, 1993),  # Next month is in the same year
        (2021, "November", 5, 2022),  # Next month is in the same year
        (2022, "December", 2, 2023),  # Next month is in the same year
        (2023, "January", 1, 2023),  # Next month is in the same year
    ]

    for year, month, spi_index, expected_forecast_year in forecast_year_tests:
        actual_forecast_year = get_forecast_year(year, month, spi_index)
        print(f"For {year}, {month} with spi_index {spi_index}, expected forecast year is {expected_forecast_year}. Actual: {actual_forecast_year}")



def create_spi_1_lead_time_data_structure(dataset_xa):
    lead_times = dataset_xa.forecast_index.values
    forecast_years = dataset_xa.year.values
    forecast_months = dataset_xa.month.values
    ensemble_member = dataset_xa.ensemble_member

    xarray = xr.DataArray(
        data=np.nan,  # Initialize with NaNs
        dims=["ensemble_member", "lead_time", "forecasted_year", "forecasted_month"],
        coords={
            'ensemble_member': copy.deepcopy(ensemble_member),
            'lead_time': copy.deepcopy(lead_times),
            'forecasted_year': copy.deepcopy(forecast_years),
            'forecasted_month': copy.deepcopy(forecast_months)
        },
        # name="SPI-1",
    )

    lead_time_dataset = xr.Dataset({"spi_1_lead_time_values": xarray})

    return lead_time_dataset



def convert_spi_1_to_lead_time(dataset_spi_1_original_xa, save=False, save_filename=None):

    all_month = dataset_spi_1_original_xa.month.values
    all_forecast_index = dataset_spi_1_original_xa.forecast_index.values

    all_year = dataset_spi_1_original_xa.year.values
    all_ensemble_member = dataset_spi_1_original_xa.ensemble_member.values

    dataset_spi_1_lead_time_xa = create_spi_1_lead_time_data_structure(dataset_spi_1_original_xa)

    # Populate the lead-time matrix
    for year in all_year:
        for month in all_month:
            for spi_index in all_forecast_index:
                for ensemble_member in all_ensemble_member:

                    # Get the SPI value
                    spi_value = float(dataset_spi_1_original_xa.sel(ensemble_member=ensemble_member, year=year, month=month, forecast_index=spi_index)['spi_1_values'])

                    # Determine the forecast month and year
                    forecasted_month, _ = get_next_month(month, spi_index)
                    forecasted_year = get_forecast_year(year, month, spi_index)

                    if forecasted_year == 2024 or forecasted_year == 2022:
                        print(f"Breaking due to forecasted year being {forecasted_year} in month {forecasted_month:<{10}} from year {year} and month {month}")
                        continue

                    dataset_spi_1_lead_time_xa['spi_1_lead_time_values'].loc[dict(ensemble_member=ensemble_member, lead_time=spi_index, forecasted_year=forecasted_year, forecasted_month=forecasted_month)] = spi_value

    if save:
        save_xarray_as_netcdf(dataset_spi_1_lead_time_xa, save_filename)

    return dataset_spi_1_lead_time_xa


#######################################################################################################################
############################################### LEAD TIME TRANSFORMATION ##############################################
#######################################################################################################################

def plot_spi_1_across_years_per_lead_time(dataset_spi_1_lead_time_xa, month='January', ensemble_member=0):
    import matplotlib.pyplot as plt

    month_em_data = dataset_spi_1_lead_time_xa.sel(forecasted_month=month, ensemble_member=ensemble_member)['spi_1_lead_time_values']
    df = month_em_data.to_dataframe().reset_index()

    # Filtering out extreme values lower than -5
    df_filtered = df[df['spi_1_lead_time_values'] > -5]

    # Plotting
    fig, ax = plt.subplots(figsize=(18, 4))  # Adjust the size as necessary

    # Define line thickness for lead times and select a colormap
    line_thickness = {1: 5, 2: 4, 3: 3, 4: 2, 5: 1, 6: 0.5}
    color_map = plt.cm.RdYlBu(np.linspace(0, 1, len(line_thickness)))

    # Loop through each unique lead time and plot, filtering out values lower than -5
    for i, lead_time in enumerate(sorted(df_filtered['lead_time'].unique())):

        plot_data = df_filtered[df_filtered['lead_time'] == lead_time]

        ax.plot(plot_data['forecasted_year'], plot_data['spi_1_lead_time_values'], label=f'Lead Time {lead_time}',
                linewidth=line_thickness[lead_time], color=color_map[i])

    ax.set_title('Ensamble member {}: Filtered SPI-1 Values Over Time for {}'.format(ensemble_member, month))
    ax.set_xlabel('Year')
    ax.set_ylabel('SPI-1 Value')

    ax.grid(True)
    ax.legend()
    plt.show()


def plot_spi_1_across_years_per_lead_time_for_months(dataset_spi_1_lead_time_xa, ensemble_member=0):
    import matplotlib.pyplot as plt

    line_thickness = {1: 5, 2: 4, 3: 3, 4: 2, 5: 1, 6: 0.5}
    color_map = plt.cm.RdYlBu(np.linspace(0, 1, len(line_thickness)))
    
    MONTHS = dataset_spi_1_lead_time_xa.forecasted_month.values

    fig, axs = plt.subplots(len(MONTHS), figsize=(18, 4*len(MONTHS)))

    for i, month in enumerate(MONTHS):
        # Filtering the dataset for the current month across all years for each lead time
        month_data = dataset_spi_1_lead_time_xa.sel(forecasted_month=month, ensemble_member=ensemble_member)['spi_1_lead_time_values']

        df = month_data.to_dataframe().reset_index()

        df_filtered = df[df['spi_1_lead_time_values'] > -5]
        # Plotting
        ax = axs[i]
        # Loop through each unique lead time and plot, filtering out values lower than -5
        for j, lead_time in enumerate(sorted(df_filtered['lead_time'].unique())):

            plot_data = df_filtered[df_filtered['lead_time'] == lead_time]
            ax.plot(plot_data['forecasted_year'], plot_data['spi_1_lead_time_values'], label=f'Lead Time {lead_time}',
                    linewidth=line_thickness[lead_time], color=color_map[j])
        
        ax.set_title('Ensemble member {}: Filtered SPI-1 Values Over Time for {}'.format(ensemble_member, month))
        ax.set_xlabel('Year')
        ax.set_ylabel('SPI-1 Value')
        
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    plt.show()



def plot_median_of_ensemble_members_spi_1_across_years_per_lead_time(dataset_spi_1_lead_time_xa, month='January'):
    import matplotlib.pyplot as plt

    line_thickness = {1: 5, 2: 4, 3: 3, 4: 2, 5: 1, 6: 0.5}
    color_map = plt.cm.RdYlBu(np.linspace(0, 1, len(line_thickness)))

    MONTHS = dataset_spi_1_lead_time_xa.forecasted_month.values

    fig, axs = plt.subplots(len(MONTHS), figsize=(18, 4*len(MONTHS)))

    for i, month in enumerate(MONTHS):
        # Filtering the dataset for the current month across all years for each lead time
        month_data = dataset_spi_1_lead_time_xa.sel(forecasted_month=month)['spi_1_lead_time_values'].median(dim='ensemble_member')

        df = month_data.to_dataframe().reset_index()
        df_filtered = df[df['spi_1_lead_time_values'] > -5]

        ax = axs[i]
        # Loop through each unique lead time and plot, filtering out values lower than -5
        for j, lead_time in enumerate(sorted(df_filtered['lead_time'].unique())):
            # Select data for the current lead time
            plot_data = df_filtered[df_filtered['lead_time'] == lead_time]

            ax.plot(plot_data['forecasted_year'], plot_data['spi_1_lead_time_values'], label=f'Lead Time {lead_time}',
                    linewidth=line_thickness[lead_time], color=color_map[j])

        ax.set_title('Median of ensemble member\'s SPI-1s: Filtered SPI-1 Values Over Time for {}'.format(month))
        ax.set_xlabel('Year')
        ax.set_ylabel('SPI-1 Value')

        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    plt.show()


def plot_ensemble_members_boxplot_across_years_for_lead_times(dataset_spi_1_lead_time_xa, month='January'):
    import matplotlib.pyplot as plt
    import seaborn as sns


    for lead_time in np.array(dataset_spi_1_lead_time_xa['lead_time']):

        ds_ensemble = dataset_spi_1_lead_time_xa.sel(lead_time=lead_time, forecasted_month=month)
        df = ds_ensemble.to_dataframe().reset_index()

        plt.figure(figsize=(12, 3))
        sns.boxplot(x='forecasted_year', y='spi_1_lead_time_values', data=df)
        plt.xlabel('Forecasted Year')
        plt.ylabel('SPI 1 Lead Time Values')
        plt.title('Lead time: {} | Forecast Month: {} | Boxplot of Ensemble Member Values for Each Year'.format(lead_time, month))
        plt.xticks(rotation=45)
        plt.show()

def plot_ensemble_members_boxplot_across_years_for_months(dataset_spi_1_lead_time_xa, lead_time=1):
    import matplotlib.pyplot as plt
    import seaborn as sns


    for month in np.array(dataset_spi_1_lead_time_xa['forecasted_month']):

        ds_ensemble = dataset_spi_1_lead_time_xa.sel(lead_time=lead_time, forecasted_month=month)
        df = ds_ensemble.to_dataframe().reset_index()

        plt.figure(figsize=(12, 3))
        sns.boxplot(x='forecasted_year', y='spi_1_lead_time_values', data=df)
        plt.xlabel('Forecasted Year')
        plt.ylabel('SPI 1 Lead Time Values')
        plt.title('Lead time: {} | Forecast Month: {} | Boxplot of Ensemble Member Values for Each Year'.format(lead_time, month))
        plt.xticks(rotation=45)
        plt.show()



#######################################################################################################################
############################################### EOBS ##############################################
#######################################################################################################################







