import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import numpy as np
import iris
import iris.coords as coords
import iris.cube as cube
import iris.coord_systems as ics
from iris_grib.grib_phenom_translation import GRIBCode
from netCDF4 import Dataset
import os
import glob

def fetch_directory(url):
    r = requests.get(url)
    if r.status_code == 200:
        return r.text
    else:
        return None


def find_latest_date(base_url, html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    dates = []

    for link in soup.find_all('a'):
        date_candidate = link.get('href').strip('/')
        if len(date_candidate) == 8 and date_candidate.isdigit():
            date_url = urljoin(base_url, date_candidate + '/WXO-DD/model_ciops/salish-sea/500m/')
            if fetch_directory(date_url):
                dates.append(date_candidate)

    return sorted(dates, reverse=True)[0] if dates else None


def find_latest_hour(base_url, latest_date):
    url = urljoin(base_url, f"{latest_date}/WXO-DD/model_ciops/salish-sea/500m/")
    directory_listing = fetch_directory(url)

    if directory_listing:
        soup = BeautifulSoup(directory_listing, 'html.parser')
        hours = [link.get('href').strip('/') for link in soup.find_all('a') if link.get('href').strip('/').isdigit()]
        return sorted(hours, reverse=True)[0] if hours else None

def convert_grib2(fdate, ftime):

    # Download the data iteratively
    for i in range(1, 49):
        fhour = str(i).zfill(3)

        path = 'https://hpfx.collab.science.gc.ca/' + fdate + '/WXO-DD/model_ciops/salish-sea/500m/' \
               + ftime + '/' + fhour + '/'

        ufile = fdate + 'T' + ftime + 'Z_MSC_CIOPS-SalishSea_SeaWaterVelocityX_DBS-0.5m_LatLon0.008x0.005_PT' + fhour + 'H.nc'
        vfile = fdate + 'T' + ftime + 'Z_MSC_CIOPS-SalishSea_SeaWaterVelocityY_DBS-0.5m_LatLon0.008x0.005_PT' + fhour + 'H.nc'

        response = requests.get(path + ufile)
        if response.status_code == 200:
            with open('u' + fhour + '.nc', 'wb') as f:
                f.write(response.content)
        else:
            print(f"Failed to download the file, status code: {response.status_code}")


        response = requests.get(path + vfile)
        if response.status_code == 200:
            with open('v' + fhour + '.nc', 'wb') as f:
                f.write(response.content)
        else:
            print(f"Failed to download the file, status code: {response.status_code}")

    U = np.zeros(shape=(48, 888, 629))
    V = np.zeros(shape=(48, 888, 629))
    dates = np.zeros(shape=(48), dtype='datetime64[s]')

    for i in range(1, 49):
        fhour = str(i).zfill(3)

        # Load the cubes from the local files
        u = Dataset('u' + fhour + '.nc', 'r')
        v = Dataset('v' + fhour + '.nc', 'r')

        U[i-1, :, :] = np.array(u['vozocrtx'])
        V[i-1, :, :] = np.array(v['vomecrty'])

        epoch = np.datetime64(u['time'].units[14:])
        dt_dt64 = epoch + np.timedelta64(int(np.array(u['time'])[0]), 's')
        dates[i-1] = dt_dt64




    # Define the bounding box with [lon_min, lon_max] and [lat_min, lat_max]
    lon_min, lon_max = -129.0, -122.0
    lat_min, lat_max = 47.0, 51.0

    # Define latitude and longitude coordinates
    new_lat = np.linspace(min(np.array(u['latitude'])), max(np.array(u['latitude'])), len(np.array(u['latitude'])))
    latitude = coords.DimCoord(new_lat, standard_name='latitude', units='degrees')

    new_lon = np.linspace(min(np.array(u['longitude'])), max(np.array(u['longitude'])), len(np.array(u['longitude'])))
    longitude = coords.DimCoord(new_lon-360, standard_name='longitude', units='degrees', circular=True)


    # Create a GeogCS (Geographic coordinate system)
    lat_lon_cs = ics.GeogCS(semi_major_axis=6371000.0)  # WGS-84 ellipsoid

    # Attach the coordinate system to existing DimCoords (assuming you have latitude and longitude coordinates)
    latitude.coord_system = lat_lon_cs
    longitude.coord_system = lat_lon_cs

    # Define time coordinate (assuming the time unit is hours since some reference time)
    # Create an array of datetime64 objects for the time coordinate
    # Starting from '2023-01-01T00:00:00' with 1-hour intervals
    time_data = dates

    # Convert to seconds since the epoch (1970-01-01 00:00:00) so Iris can understand it
    epoch = np.datetime64('1970-01-01T00:00:00')
    time_data_seconds = (time_data - epoch) / np.timedelta64(1, 's')

    # Create time coordinate

    time = coords.DimCoord(time_data_seconds, standard_name='time', units='seconds since 1970-01-01 00:00:00')

    # Create Iris cubes for current speed and direction
    gU = GRIBCode(2, 10, 1, 2)
    gV = GRIBCode(2, 10, 1, 3)

    # create the cubes
    U_cube = cube.Cube(
        U,
        standard_name=None,
        long_name='ocean_surface_current_U',
        units='m s-1',
        attributes={'GRIB_PARAM': gU},
        dim_coords_and_dims=[(time, 0), (latitude, 1), (longitude, 2)]
    )

    V_cube = cube.Cube(
        V,
        standard_name=None,
        long_name='ocean_surface_current_V',
        units='m s-1',
        attributes={'GRIB_PARAM': gV},
        dim_coords_and_dims=[(time, 0), (latitude, 1), (longitude, 2)]
    )

    # deal with NaN values
    U_cube.data = np.ma.masked_where(U_cube.data > 1e19, U_cube.data)
    V_cube.data = np.ma.masked_where(V_cube.data > 1e19, V_cube.data)

    # add a forecast_period (required by GRIB2 writer)
    U_cube.add_aux_coord(iris.coords.DimCoord(0, standard_name='forecast_period', units='hours'))
    V_cube.add_aux_coord(iris.coords.DimCoord(0, standard_name='forecast_period', units='hours'))

    # add a vertical coordinate (required by GRIB2 writer)
    U_cube.add_aux_coord(iris.coords.DimCoord(0, "height", units="m"))
    V_cube.add_aux_coord(iris.coords.DimCoord(0, "height", units="m"))

    # Create a cube list to hold both cubes
    cube_list = iris.cube.CubeList([U_cube, V_cube])

    iris.save(cube_list, '/Users/chrisbradley/Downloads/ss500_' + fdate + '_' + ftime + '.grib2', saver='grib2')

    # Clean Up
    nc_files = glob.glob("*.nc")
    for filename in nc_files:
        try:
            os.remove(filename)
        except Exception as e:
            print(f"Could not delete {filename}: {e}")

def get_recent():
    base_url = 'https://hpfx.collab.science.gc.ca/'
    directory_listing = fetch_directory(base_url)

    if directory_listing:
        latest_date = find_latest_date(base_url, directory_listing)

        if latest_date:
            # print(f"The most recent date is {latest_date}.")
            latest_hour = find_latest_hour(base_url, latest_date)

            if latest_hour:
                # print(f"The most recent hour is {latest_hour}.")
                convert_grib2(latest_date, latest_hour)
            else:
                print("Could not find any hour folders.")
        else:
            print("Could not find any date folders.")
    else:
        print("Failed to fetch the directory listing.")



