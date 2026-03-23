# Extended Range Drought Forecasting

This repository contains the code for my Master's thesis on extended-range drought forecasting. The project investigates the potential of machine learning to improve drought forecasts over the Nordic region by combining dynamical seasonal forecasts with historical observations.

The models predict two specific drought indices at a one-month lead time:
* **SPI-1**: 1-month Standardized Precipitation Index
* **WDF**: Wet Day Fraction

## Data Sources

The project relies on two primary datasets:
* **ECMWF SEAS5**: Dynamical ensemble seasonal forecasts (providing variables like temperature, precipitation rate, and mean sea level pressure).
* **E-OBS**: Historical gridded observational data (providing mean temperature, precipitation, and sea level pressure).

## Project Structure

The codebase is structured around two identical analytical pipelines based on the geographic region:

1. **Eastern Norway (`EastNor_`)**: A pilot region used for extensive data exploration, feature engineering, and model selection.
2. **Nordic Region / Fennoscandia (`Nordic_`)**: The full study area (mainland Norway, Finland, Sweden, and Denmark) where the optimal models identified in the pilot are scaled up and deployed.

Shared `.py` files contain common functions and utilities utilized by both regional pipelines.

## Analysis Pipeline

The Jupyter Notebooks are numbered sequentially (1 through 7) to reflect the order of execution. 

### 1. Data Processing and Cleaning
* **Files**: `*_1_ECMWF_rr_tg_pp_process.ipynb`, `*_1_EOBS_rr_tg_pp_process.ipynb`, `*_1_EOBS_spi_process.ipynb`
* **Description**: Processes raw NetCDF files from ECMWF and E-OBS. Extracts meteorological variables, handles grid subsetting, and calculates the SPI-1 and WDF from daily E-OBS precipitation data.

### 2. Data Synthesis
* **File**: `*_2_EOBS_rr_tg_pp_spi_merge.ipynb`
* **Description**: Synthesizes the independently processed E-OBS variables (pressure, precipitation, temperature, WDF, and SPI) into a unified dataset.

### 3. Feature Engineering
* **File**: `*_3_EOBS_ECMWF_merge_season.ipynb`
* **Description**: Aligns the ECMWF forecast data with E-OBS observational data. Adds seasonal indicators and lagged variables required for the predictive models.

### 4. Mean Modeling and Analysis
* **Files**: `*_4_merged_analysis.ipynb`, `*_4_mean_modelling.ipynb` (or `*_4_model_runs.ipynb`)
* **Description**: Conducts exploratory data analysis and correlation mapping. Trains and evaluates multiple predictive models (e.g., linear models, multilayer perceptrons) using Leave-One-Out Cross-Validation.

### 5. Model Selection
* **File**: `*_5_model_selection_for_nordicregion.ipynb` (or `fennoscandia`)
* **Description**: Evaluates the performance of the models tested in step 4. Provides statistical justification for selecting the most parsimonious and effective models to deploy on the larger Nordic region.

### 6. Error Relationship Analysis
* **File**: `*_6_error_relationship.ipynb`
* **Description**: Investigates the relationship between the mean prediction errors and the input variables to determine if forecast uncertainty can be systematically predicted.

### 7. Uncertainty Quantification
* **Files**: `*_7_uncertainty_quantification_SPI.ipynb`, `*_7_uncertainty_quantification_WDF.ipynb`
* **Description**: Builds secondary models to quantify the uncertainty of the predictions, generating the final probabilistic forecasts for both SPI-1 and WDF.

## Getting Started

To reproduce the results:

1. Place the raw E-OBS and ECMWF NetCDF datasets in the expected raw data directories.
2. Ensure you have the necessary Python libraries installed (e.g., `numpy`, `pandas`, `xarray`, `scikit-learn`, `cartopy`).
3. Run the notebooks in numerical order. It is recommended to run the Eastern Norway (`EastNor_`) pilot pipeline first to understand the data processing and model selection logic before running the computationally heavier Nordic region pipeline.
