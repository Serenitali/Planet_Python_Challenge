'''
Author: Serena Annibali

Description
This script is a first solution to a potential bigger challenge that evaluates
weather conditions for satellite observation of areas of interest (AOI),
which are assumed to be cities and towns of higher population.
It represents a decision making process for the immediate acquisition
of satellite optical images, according to actual weather conditions, 
specifically where conditions of clear sky are met.
For semplicity, the conditions are requested at the instantaneaous timestamp.
For this case, current weather conditions are obtained through API request to 
OpenWeatherMap https://openweathermap.org/current, making use of a free account.

    Note about the solution: A more extended analysis can be done to obtain the
    average cloud coverage over extended ranges of time, i.e.  looking at the 
    historical (https://openweathermap.org/history) and the forecasted (16 days
    forecast, https://openweathermap.org/forecast16) data, through API request 
    to OpenWeatherMap making use of a paid account. 


Input data
    data_populated_places: folder  
        It contains
         the data of the most populated city and town as shp file 
        (ne_10m_populated_places_simple.shp). Provided source: https://www.naturalearthdata.com/downloads/10m-cultural-vectors/10m-populated-places/
         the list of city ID in a json file (city.list.json). Source: http://bulk.openweathermap.org/sample/
        
Output
    Pandas Dataframe
        It contains the selected cities/locations and their weather condition as 
        percentage cloud coverage.
    Plots
        Visualisation of the results, using the function plot_on_map
        

The function selected_cities has to receive as arguments: 
    selection
        it is the number of most populated cities.
    appid
        it is the ID account for the API request.
It returns the selected cities/locations and the weather condition as percentage
cloud coverage. For the API request it calls the function get_weather.
'''

# Import the needed modules
import numpy as np
import pandas as pd
import requests
import time
import geopandas as gpd
import matplotlib.pyplot as plt


def plot_on_map(dataset, title):
    '''
    This function plots the weather attribute (specified by column) from a
    given dataset (specified by dataset) of geolocation, and visualise it on
    a world map.
    '''
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    base = world.plot(color='white', edgecolor='black')
    dataset.plot(ax=base, column='clouds', legend=True)
    plt.xlabel('Longitude °')
    plt.ylabel('Latitude °')
    plt.title(title)
    plt.show()


def get_weather(API_URL, id_list):
    '''
    This function requires as arguments the endpoint and the list of cities IDs
    and returns the weather attributes ('main' and 'description'), the cloud 
    coverage and the time stamp.
    '''
    # Create empty lists to store the content of the API request
    weath_main = []
    weath_desc = []
    cloud_cov = []
    # For the API request, it would be possible to use the following endpoint:
    # http://api.openweathermap.org/data/2.5/group?id={}&units=metric
    # (where id is a concatenated list of csv ids) and retrieve the data for 20
    # cities in only one request.
    # However, in case one city ID is not found, the entire request fails and
    # the case of error needs to be handled in a different way. For instance,
    # making use of the try: except:
    # then removing the only failed ID cities and process the request again.
    for city_id in id_list:
        url = API_URL.format(city_id)
        # Create a response object to store the request response
        response = requests.get(url)
        # Convert the response content into a JSON format
        resp_j = response.json()
        # Check that the request has succeeded, and assign the returned content
        if response.status_code == 200:
            weath_main.append(
                resp_j['weather'][0]['main'])  # 'weather' is a list
            weath_desc.append(resp_j['weather'][0]['main'])
            cloud_cov.append(resp_j['clouds']['all'])
        # If it has not succeeded, print an error message, and assign Nan values
        # in order to respect the length of the dataframe
        else:
            print('Request not successfull for city_id {}'.format(city_id))
            print(response.status_code, response.text)
            weath_main.append(np.nan)
            weath_desc.append(np.nan)
            cloud_cov.append(np.nan)
            
    return cloud_cov, weath_desc, weath_main, resp_j['dt']


def select_cities(selection, appid):
    '''
    This function select the cities and make the API request, calling the 
    function get_weather. It returns the selected cities and plot the corresponding
    cloud coverage.
    '''
    # Set the path where the data are stored
    path_data = 'data_populated_places/ne_10m_populated_places_simple.shp'
    path_cityid = 'data_populated_places/city.list.json'

    # Load the data of the populated cities and towns and the corresponding IDs
    # as geopandas and pandas dataframes
    cities_data = gpd.read_file(path_data)
    cities_id = pd.read_json(path_cityid)

    # Sort the features (i.e. most populated cities and towns)
    # according to the attribute 'pop_max'
    sorted_cities = cities_data.sort_values('pop_max', ascending=False)
    # Add the ID attribute where the same city name is found, using the
    # attribute 'name_ascii'
    fulldataset = sorted_cities.merge(cities_id, how='inner',
                                      left_on='nameascii', right_on='name')

    # Check the country code (column 'iso_a2' in sorted_cities, 'country' in
    # 'cities_id') of the merged features
    full_dataset = fulldataset[fulldataset['iso_a2'] == fulldataset['country']]
    # and clear the full_dataset from duplicates
    clear_dataset = full_dataset.drop_duplicates(subset='nameascii',
                                                 keep='first')

    # Select a small number of features, e.g. 50 as requested
    sel_dataset = clear_dataset[0:selection]

    # Prepare the list of cities ID for the API request
    id_list = list(sel_dataset['id'])

    # Prepare the API request: assign the URL. appid 'ac36bd53beae1a124b13dead05d3dd5e'
    api_path = 'http://api.openweathermap.org/data/2.5/weather?id={}&APPID='
    API_URL = api_path + appid

    # Call the function get_weather to make the API request.
    cloud_cov, weath_desc, weath_main, resp_time = get_weather(API_URL, id_list)
    #  Convert the timestamp of the request from unix format to a string
    timestr = time.strftime("%D %H:%M",
                            time.localtime(int(format(resp_time))))

    sel_dataset.loc[:, 'weather'] = weath_main
    sel_dataset.loc[:, 'weather_description'] = weath_desc
    sel_dataset.loc[:, 'clouds'] = cloud_cov
    # assign No data to avoid problem with Nan values when plotting
    sel_dataset.fillna('No data', inplace=True)

    # Visualise the cloud coverage of the selected dataset on a map,
    # calling the module plot_on_map
    title_clouds ='Cloud % coverage of {} most populated cities'.format(selection)
    plot_on_map(sel_dataset, title_clouds)

    # Selection criteria: Clear sky for acquisition of optimal imaging data by
    # satellite
    final_dataset = sel_dataset.loc[sel_dataset['weather'] == 'Clear']

    # Visualisation of the selected data on a map
    title_clear = 'Location of selected clear sky city on {}'.format(timestr)
    plot_on_map(final_dataset, title_clear)

    return final_dataset[['nameascii', 'clouds']]


if __name__ == '__main__':
    selected_cities = select_cities(selection=50,
                                    appid='ac36bd53beae1a124b13dead05d3dd5e')


