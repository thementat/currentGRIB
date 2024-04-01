import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import os
import tempfile
import numpy as np
from netCDF4 import Dataset

def fetch_directory(url):
    r = requests.get(url)
    if r.status_code == 200:
        return r.text
    else:
        return None

def find_latest_datetime():
    url = 'https://hpfx.collab.science.gc.ca/today/model_ciops/salish-sea/500m/'
    directory_listing = fetch_directory(url)
    soup = BeautifulSoup(directory_listing, 'html.parser')
    hours = [link.get('href').strip('/') for link in soup.find_all('a') if link.get('href').strip('/').isdigit()]

    datetimes = []

    for hour in hours:
        fileurl = urljoin(url, hour + '/001')
        r = requests.get(fileurl)
        soup = BeautifulSoup(r.text, 'html.parser')
        rows = [link.get('href').strip('/') for link in soup.find_all('a') if link.get('href').strip('/')]

        for row in rows:
            if row[-3:] == '.nc':
                datetimes.append(row[0:12])
                break

    return sorted(datetimes)[-1]

@app.route(route="ss500m")
def ss500m(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    date = req.params.get('date')
    if not date:
        try:
            req_body = req.get_json()
            if isinstance(req_body, str):  # Check if req_body is a string
                req_body = json.loads(req_body)  # Parse string to dict
                date = req_body.get('date')
        except ValueError:
            pass
        else:
            date = req_body.get('date')

    dt = find_latest_datetime()
    tdate = dt[0:8]
    ttime = dt[9:11]

    U = np.zeros(shape=(48, 888, 629))
    V = np.zeros(shape=(48, 888, 629))
    dates = np.zeros(shape=(48), dtype='datetime64[s]')

    for i in range(1, 49):
        thour = str(i).zfill(3)
        path = f'https://hpfx.collab.science.gc.ca/today/model_ciops/salish-sea/500m/{ttime}/{thour}/'
        ufile = f'{dt}_MSC_CIOPS-SalishSea_SeaWaterVelocityX_DBS-0.5m_LatLon0.008x0.005_PT{thour}H.nc'
        vfile = f'{dt}_MSC_CIOPS-SalishSea_SeaWaterVelocityY_DBS-0.5m_LatLon0.008x0.005_PT{thour}H.nc'

        # Process U file
        response = requests.get(path + ufile, stream=True)
        if response.status_code == 200:
            with tempfile.NamedTemporaryFile(delete=True) as tmp_u:
                for chunk in response.iter_content(chunk_size=128):
                    tmp_u.write(chunk)
                tmp_u.flush()
                u = Dataset(tmp_u.name, 'r')
                U[i - 1, :, :] = np.array(u['vozocrtx'])
                u.close()
        else:
            print(f"Failed to download {ufile}, status code: {response.status_code}")

        # Process V file
        response = requests.get(path + vfile, stream=True)
        if response.status_code == 200:
            with tempfile.NamedTemporaryFile(delete=True) as tmp_v:
                for chunk in response.iter_content(chunk_size=128):
                    tmp_v.write(chunk)
                tmp_v.flush()
                v = Dataset(tmp_v.name, 'r')
                V[i - 1, :, :] = np.array(v['vomecrty'])
                v.close()
        else:
            print(f"Failed to download {vfile}, status code: {response.status_code}")

        epoch = np.datetime64(u['time'].units[14:])
        dt_dt64 = epoch + np.timedelta64(int(np.array(u['time'])[0]), 's')
        dates[i - 1] = dt_dt64

    # Define the bounding box with [lon_min, lon_max] and [lat_min, lat_max]
    lon_min, lon_max = -129.0, -122.0
    lat_min, lat_max = 47.0, 51.0

    # Define latitude and longitude coordinates
    new_lat = np.linspace(min(np.array(u['latitude'])), max(np.array(u['latitude'])), len(np.array(u['latitude'])))
    latitude = coords.DimCoord(new_lat, standard_name='latitude', units='degrees')

    new_lon = np.linspace(min(np.array(u['longitude'])), max(np.array(u['longitude'])),
                          len(np.array(u['longitude'])))
    longitude = coords.DimCoord(new_lon - 360, standard_name='longitude', units='degrees', circular=True)

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

    file_name = 'ss500m.' + tdate + '.' + ttime + '.grib2'

    # Create a temporary file to save the data
    with tempfile.NamedTemporaryFile(mode='w+b', suffix='.grib2', delete=False) as tmp_file:
        # Use iris to save the cube data to the temporary file
        iris.save(cube_list, tmp_file.name, saver='grib2')

        # Seek to the beginning of the file to read its content
        tmp_file.seek(0)
        file_content = tmp_file.read()

    headers = {
        'Content-Disposition': f'attachment; filename="{file_name}"'
    }

    # Return the file content in the HttpResponse
    return func.HttpResponse(body=file_content, headers=headers, status_code=200, mimetype='application/octet-stream')