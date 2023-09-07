import h5py
import numpy as np
import xarray as xr
# import eccodes
# from cfgrib.xarray_to_grib import to_grib
from scipy import interpolate
# import pygrib
import iris
import iris.coords as coords
import iris.cube as cube
import iris.coord_systems as ics
from iris_grib.grib_phenom_translation import GRIBCode
import requests
from netCDF4 import Dataset






def knots_to_ms(knots):
    """Convert knots to meters per second."""
    return knots * 0.514444


def wind_components(speed_in_knots, direction_in_degrees):
    """
    Convert wind speed (in knots) and direction (in degrees) to U and V components in m/s.

    Parameters:
        speed_in_knots (float or np.array): Wind speed in knots.
        direction_in_degrees (float or np.array): Wind direction in degrees. 0 degrees is from the North.

    Returns:
        tuple: U and V wind components in m/s.
    """
    speed_in_ms = knots_to_ms(speed_in_knots)
    direction_rad = np.radians(direction_in_degrees)

    U = -speed_in_ms * np.sin(direction_rad)
    V = -speed_in_ms * np.cos(direction_rad)

    return U, V


def ss500_grib2(fdate, ftime):


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

        d_64 = np.datetime64(fdate[0:4] + '-' + fdate[4:6] + '-' + fdate[6:8])
        dt_dt64 = d_64 + np.timedelta64(i-1, 'h')
        dates[i-1] = dt_dt64


    v_cube = v_cube = iris.load('temporary_file.nc')

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

    iris.save(cube_list, '/Users/chrisbradley/Downloads/output.grib2', saver='grib2')










def noaa_grib2():



    infile = '/Users/chrisbradley/Downloads/111US00_WCOFS_20230824T03Z.h5'
    outfile = '/Users/chrisbradley/Downloads/output.grib'
    lastoutfile = '/Users/chrisbradley/Downloads/f_output.grib'

    # Open the HDF5 file
    hdf5_file = h5py.File(infile, 'r')

    ndates = len(hdf5_file['SurfaceCurrent']['SurfaceCurrent.01']) - 2



    # Define the bounding box with [lon_min, lon_max] and [lat_min, lat_max]
    lon_min, lon_max = -129.0, -122.0
    lat_min, lat_max = 47.0, 51.0

    # Define the value for nulls
    nullval = -99

    # Step 1: Read HDF5 Data
    original_data = hdf5_file['SurfaceCurrent']['SurfaceCurrent.01']['Group_001']['values'][:]

    # Step 2: Understand the Current Grid
    # original_lon and original_lat are random arrays for this example
    original_lat = hdf5_file['SurfaceCurrent']['SurfaceCurrent.01']['Positioning']['geometryValues']['latitude']
    original_lon = hdf5_file['SurfaceCurrent']['SurfaceCurrent.01']['Positioning']['geometryValues']['longitude']

    # Filter points based on bounding box
    mask_lon = (original_lon >= lon_min) & (original_lon <= lon_max)
    mask_lat = (original_lat >= lat_min) & (original_lat <= lat_max)
    mask = mask_lon & mask_lat

    # Filter data and coordinates
    filtered_lon = original_lon[mask]
    filtered_lat = original_lat[mask]
    filtered_data = original_data[mask]

    # Create coordinates for interpolation
    coordinates = np.array([(lon, lat) for lon, lat in zip(filtered_lon, filtered_lat)])

    # Step 3: Specify New Grid
    lon_space = 0.0075
    lat_space = 0.005
    new_lon = np.linspace(lon_min, lon_max, int((lon_max-lon_min)/lon_space))
    new_lat = np.linspace(lat_min, lat_max, int((lat_max-lat_min)/lat_space))
    new_lon_grid, new_lat_grid = np.meshgrid(new_lon, new_lat)
    new_coordinates = np.c_[new_lon_grid.ravel(), new_lat_grid.ravel()]



    # Step 4: Interpolation


    U = np.zeros(shape=(ndates,len(new_lat),len(new_lon)))
    V = np.zeros(shape=(ndates,len(new_lat),len(new_lon)))
    dates = np.zeros(shape=(ndates), dtype='datetime64[s]')

    for index, group in enumerate(hdf5_file['SurfaceCurrent']['SurfaceCurrent.01']):
        if group != 'uncertainty' and group != 'Positioning':
            print(group)
            g = hdf5_file['SurfaceCurrent']['SurfaceCurrent.01'][group]
            dt_str = g.attrs['timePoint']
            dt_dt64 = np.datetime64(dt_str[0:4] + '-' + dt_str[4:6] + '-' + dt_str[6:8] \
                + dt_str[8:11] + ':' + dt_str[11:13] + ':' + dt_str[13:15])

            # Step 4: Interpolation
            new_spd = interpolate.griddata(coordinates, g['values'][mask]['surfaceCurrentSpeed'], new_coordinates,
                                           method='cubic')
            new_dir = interpolate.griddata(coordinates, g['values'][mask]['surfaceCurrentDirection'], new_coordinates,
                                           method='cubic')

            # Step 5: Convert to vectors

            U[index, :, :], V[index, :, :] = wind_components(new_spd.reshape(new_lon_grid.shape), new_dir.reshape(new_lon_grid.shape))
            dates[index] = dt_dt64

    # Define latitude and longitude coordinates
    latitude = coords.DimCoord(new_lat, standard_name='latitude', units='degrees')
    longitude = coords.DimCoord(new_lon, standard_name='longitude', units='degrees', circular=True)

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

    # Convert NULL values
    # surfaceCurrentSpeed = np.nan_to_num(surfaceCurrentSpeed, nullval)
    # surfaceCurrentDirection = np.nan_to_num(surfaceCurrentDirection, nullval)

    # create the cubes
    U_cube = cube.Cube(
        U,
        standard_name=None,
        long_name='ocean_surface_current_speed',
        units='m s-1',
        attributes={'GRIB_PARAM': gU},
        dim_coords_and_dims=[(time, 0), (latitude, 1), (longitude, 2)]
    )

    V_cube = cube.Cube(
        V,
        standard_name=None,
        long_name='ocean_surface_current_direction',
        units='degrees',
        attributes={'GRIB_PARAM': gV},
        dim_coords_and_dims=[(time, 0), (latitude, 1), (longitude, 2)]
    )

    # deal with NaN values
    U_cube.data=np.ma.masked_invalid(U_cube.data)
    V_cube.data=np.ma.masked_invalid(V_cube.data)

    # add a forecast_period (required by GRIB2 writer)
    U_cube.add_aux_coord(iris.coords.DimCoord(0, standard_name='forecast_period', units='hours'))
    V_cube.add_aux_coord(iris.coords.DimCoord(0, standard_name='forecast_period', units='hours'))

    # add a vertical coordinate (required by GRIB2 writer)
    U_cube.add_aux_coord(iris.coords.DimCoord(0, "height", units="m"))
    V_cube.add_aux_coord(iris.coords.DimCoord(0, "height", units="m"))

    # Create a cube list to hold both cubes
    cube_list = iris.cube.CubeList([U_cube, V_cube])

    iris.save(cube_list, outfile, saver='grib2')
