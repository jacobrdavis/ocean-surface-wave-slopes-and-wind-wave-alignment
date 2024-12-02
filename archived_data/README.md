# Data for: Ocean surface wave slopes and wind-wave alignment observed in Hurricane Idalia

[https://doi.org/10.5061/dryad.zw3r228h7](https://doi.org/10.5061/dryad.zw3r228h7)

## Description of the data and file structure

This dataset accompanies the article "Ocean surface wave slopes and wind-wave alignment observed in Hurricane Idalia" and  contains observational wave data and modeled wind data.  Wave measurements are collected by free-drifting microSWIFT buoys (APL-UW) and Spotter buoys (Sofar Ocean) which use GPS-derived motions to report hourly records of surface wave statistics in the form of scalar energy spectra and directional moments. The observational data are combined with modeled surface winds from the U.S. Naval Research Laboratory’s Coupled Ocean-Atmosphere Mesoscale Prediction System for Tropical Cyclones (COAMPS-TC) which are interpolated onto the buoy observations in time and space.  Supporting data include best track data from the International Best Track Archive for Climate Stewardship (IBTrACS) dataset and bathymetry data from the General Bathymetric Chart of the Oceans (GEBCO) 2023 grid.  The data are located offshore of the Big Bend region of Florida and span approximately 00Z on 29 August to 00Z on 31 August, as Idalia was making landfall along the Florida coast.

More information on the microSWIFT and Spotter  buoys can be found in:

Thomson, J., Bush, P., Castillo Contreras, V., Clemett, N., Davis, J., De Klerk, A., Iseley, E., Rainville, E. J., Salmi, B., & Talbert, J. (2023). Development and testing of microSWIFT expendable wave buoys. *Coastal Engineering Journal*, 1–13. [https://doi.org/10.1080/21664250.2023.2283325](https://doi.org/10.1080/21664250.2023.2283325)

Raghukumar, K., Chang, G., Spada, F., Jones, C., Janssen, T., & Gans, A. (2019). Performance Characteristics of “Spotter,” a Newly Developed Real-Time Wave Measurement Buoy. *Journal of Atmospheric and Oceanic Technology*, *36*(6), 1127–1141. [https://doi.org/10.1175/JTECH-D-18-0151.1](https://doi.org/10.1175/JTECH-D-18-0151.1)

(See also: [https://www.sofarocean.com/](https://www.sofarocean.com/))

This data was collected as part of the NOPP Hurricane Coastal Impacts (NHCI) project.  More information on the NHCI project is available at: [https://nopphurricane.sofarocean.com/](https://nopphurricane.sofarocean.com/).

### Files and variables

Output data files are in netCDF format (.nc) and follow the Climate and Forecast (CF) Metadata Conventions 1.11 (CF-1.11) and Attribute Convention for Data Discovery 1-3 (ACDD-1.3).  Variable attributes describe the standard name, long name, description, units, and coverage content type of each variable.  Where applicable, standard names are from the CF Standard Name Table Version 85 (21 May 2024), otherwise this attribute is left empty.  Null floating point values are indicated by NaN and null strings are indicated by an empty string.  Data for each buoy type (microSWIFT or Spotter) are organized as single trajectory data indexed by observation time.  Each netCDF file is named according to the buoy type and ID as: "hurricane_idalia_{`buoy_type`}_{`buoy_id`}.nc".

Input data, required to run the notebooks in the code repository (accessible via GitHub, see next section), are provided in an `input_data.zip` file as Supplemental Files.  These files **are not recommended for use** **outside the notebooks** (the output files above are quality controlled, adjusted for the Doppler shift, contain more variables, and are self-documenting).

The supplemental input data  folder contains the following files:

* `hurricane_idalia_drifter_data_v3.pickle` - Original Spotter and MicroSWIFT drifter (buoy) data in Hurricane Idalia from 2023-08-28 to 2023-08-31.
* `coamps_idalia_reforecast.nc` - COAMPS-TC "reforecast" wind and pressure fields for Hurricane Idalia from 2023-08-29 to 2023-08-31. Originally accessed from MetGet on 1 November 2024.
* `al102023_best_track/AL102023_pts*` , `al102023_best_track/AL102023_lin*`, `al102023_best_track/AL102023_radii*, al102023_best_track/AL102023_windswath*` - NHC shape files of best track points, line, radii, and wind swath. Each component should include the `.shp` file and all supporting files (`.dbf`, `.prj`, `.shp.xml`, `.shx`).  These data were originally accessed from [https://www.nhc.noaa.gov/data/tcr/index.php?season=2023&basin=atl](https://www.nhc.noaa.gov/data/tcr/index.php?season=2023&basin=atl) on 11 November 2024. These data are used in accordance with the NOAA/NWS Data and Products disclaimer at: [https://www.weather.gov/disclaimer/](https://www.weather.gov/disclaimer/).
* `ibtracs.last3years.list.v04r01.csv` - International Best Track Archive for Climate Stewardship (IBTrACS) tabular data from the last 3 years (used for storm metrics). These data were originally accessed from [https://www.ncei.noaa.gov/products/international-best-track-archive](https://www.ncei.noaa.gov/products/international-best-track-archive) on 7 November 2024. Terms of use are described under the "Support" tab at: [https://www.ncei.noaa.gov/products/international-best-track-archive](https://www.ncei.noaa.gov/products/international-best-track-archive). (See citation under **Access information** below.)
* `gebco_2023_n31.0_s25.0_w-87.0_e-81.0.nc` - Gridded bathymetry data from the General Bathymetric Chart of the Oceans (GEBCO) GEBCO_2023 grid. These data were originally accessed from [https://www.gebco.net/data_and_products/gridded_bathymetry_data/](https://www.gebco.net/data_and_products/gridded_bathymetry_data/) on November 28, 2023. Terms of use are described under the "Terms of use and disclaimer" header at: [https://www.gebco.net/data_and_products/gridded_bathymetry_data/gebco_2023/](https://www.gebco.net/data_and_products/gridded_bathymetry_data/gebco_2023/). (See citation under **Access information** below.)
* `noaa_met_data/20230829I1_A.nc`, `noaa_met_data/20230829I2_A.nc` - NOAA Stepped Frequency Microwave Radiometer data collected in Hurricane Idalia on 2023-08-29. These data were originally accessed from [https://www.aoml.noaa.gov/2023-hurricane-field-program-data/](https://www.aoml.noaa.gov/2023-hurricane-field-program-data/) on 28 October 2024. These data are used in accordance with the NOAA/NWS Data and Products disclaimer at: [https://www.weather.gov/disclaimer/](https://www.weather.gov/disclaimer/).
* `usafr_met_data/20230829U1.01.txt`, `usafr_met_data/20230830U1.01.txt`, `usafr_met_data/20230831U1.01.txt` - United States Air Force Reserve 53rd WRS Stepped Frequency Microwave Radiometer data collected in Hurricane Idalia 2023-08-29 to 2023-08-31. These data were originally accessed from [https://www.aoml.noaa.gov/2023-hurricane-field-program-data/](https://www.aoml.noaa.gov/2023-hurricane-field-program-data/) on 28 October 2024. These data are used in accordance with the NOAA/NWS Data and Products disclaimer at: [https://www.weather.gov/disclaimer/](https://www.weather.gov/disclaimer/).
* `NDBC_42036_202308_D7_v00.nc` - National Data Buoy Center Station 42036 (West Tampa) data for the month of 2023-08. These data were originally accessed from [https://www.ncei.noaa.gov/archive/accession/NDBC-CMANWx](https://www.ncei.noaa.gov/archive/accession/NDBC-CMANWx) on 13 November 2023.  These data are used in accordance with the NOAA/NWS Data and Products disclaimer at: [https://www.weather.gov/disclaimer/](https://www.weather.gov/disclaimer/) and  "Use Limitations" under the "Description" tab at: [https://www.ncei.noaa.gov/archive/accession/NDBC-CMANWx](https://www.ncei.noaa.gov/archive/accession/NDBC-CMANWx). (See citation under **Access information** below.)

  Additional input file documentation is provided in `input_data/documentation/`.

## Code/software

Source code is organized into Python notebooks and can be accessed via GitHub  at [https://github.com/jacobrdavis/ocean-surface-wave-slopes-and-wind-wave-alignment-observed-in-Hurricane-Idalia/](https://github.com/jacobrdavis/ocean-surface-wave-slopes-and-wind-wave-alignment-observed-in-Hurricane-Idalia/) or via the Zenodo archive at [https://doi.org/10.5281/zenodo.13953570](https://doi.org/10.5281/zenodo.13953570).

The output files can be opened using any software that supports  netCDF .nc files (e.g., the Xarray and netCDF4 Python packages or MATLAB's ncread function).  See [https://www.unidata.ucar.edu/software/netcdf/](https://www.unidata.ucar.edu/software/netcdf/) for more information.

## Access information

Other publicly accessible locations of the data:

* An archive of microSWIFT ocean surface wave data is available as: Thomson, Jim (2024). microSWIFT ocean surface wave data [Dataset]. Dryad. [https://doi.org/10.5061/dryad.jdfn2z3j1](https://doi.org/10.5061/dryad.jdfn2z3j1)
* A publicly accessible repository of historical Spotter data is available at: [https://www.sofarocean.com/mx/sofar-spotter-archive](https://www.sofarocean.com/mx/sofar-spotter-archive)

Supporting data was derived from the following sources:

* Best track data: Gahtan, Jennifer; Knapp, Kenneth R.; Schreck, Carl J. III; Diamond, Howard J.; Kossin, James P.; Kruk, Michael C. (2024). International Best Track Archive for Climate Stewardship (IBTrACS) Project, Version 4.01. [last 3 years]. NOAA National Centers for Environmental Information. [https://doi.org/10.25921/82ty-9e16](https://doi.org/10.25921/82ty-9e16). Accessed 7 November 2024.
* Bathymetry data:  GEBCO Bathymetric Compilation Group. (2023). The GEBCO_2023 grid - a continuous terrain model of the global oceans and land. NERC EDS British Oceanographic Data Centre NOC. doi:10.5285/f98b053b-0cbc-6c23-e053-6c86abc0af7b. Accessed November 28, 2023.
* NDBC data: NOAA National Data Buoy Center (1971). Meteorological and oceanographic data collected from the National Data Buoy Center Coastal-Marine Automated Network (C-MAN) and moored (weather) buoys. [Station 42036 (LLNR 855)1028 - WEST TAMPA]. NOAA National Centers for Environmental Information. Dataset. [https://www.ncei.noaa.gov/archive/accession/NDBC-CMANWx](https://www.ncei.noaa.gov/archive/accession/NDBC-CMANWx). Accessed 13 November 2023.

