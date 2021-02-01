#!/usr/bin/env python
# coding: utf-8

# <h1 align=center><font size = 5>Segmenting and Clustering Neighborhoods in Toronto</font></h1>

# ## Introduction
# 
# For this assignment, you will be required to explore and cluster the neighborhoods in Toronto.
# 
# Start by creating a new Notebook for this assignment.
# Use the Notebook to build the code to scrape the following Wikipedia page, https://en.wikipedia.org/wiki/List_of_postal_codes_of_Canada:_M, in order to obtain the data that is in the table of postal codes and to transform the data into a pandas dataframe like the one shown below:

#  To create the dataframe:
# 
# 1. The dataframe will consist of three columns: PostalCode, Borough, and Neighborhood
# 2. Only process the cells that have an assigned borough. Ignore cells with a borough that is Not assigned.
# 3. More than one neighborhood can exist in one postal code area. For example, in the table on the Wikipedia page, you will notice that M5A is listed twice and has two neighborhoods: Harbourfront and Regent Park. These two rows will be combined into one row with the neighborhoods separated with a comma as shown in row 11 in the above table.
# 4. If a cell has a borough but a Not assigned neighborhood, then the neighborhood will be the same as the borough.
# 5. Clean your Notebook and add Markdown cells to explain your work and any assumptions you are making.
# 6. In the last cell of your notebook, use the .shape method to print the number of rows of your dataframe.
# 

# In[166]:


from bs4 import BeautifulSoup 
import requests # library to handle requests

import numpy as np # library to handle data in a vectorized manner
import pandas as pd # library for data analsysis

import json # library to handle JSON files

from geopy.geocoders import Nominatim # convert an address into latitude and longitude values
from pandas.io.json import json_normalize # tranform JSON file into a pandas dataframe

# Matplotlib and associated plotting modules
import matplotlib.cm as cm
import matplotlib.colors as colors

# import k-means from clustering stage
from sklearn.cluster import KMeans

get_ipython().system(' pip install folium==0.5.0')
import folium

print('Libraries imported.')


# Use the using the BeautifulSoup package to scrape the table from the following Wikipedia page, https://en.wikipedia.org/wiki/List_of_postal_codes_of_Canada:_M and transform the data into pandas dataframe with data cleaning 

# In[167]:


url = "https://en.wikipedia.org/wiki/List_of_postal_codes_of_Canada:_M"
html_content = requests.get(url).text
soup = BeautifulSoup(html_content, "lxml")

# find the table 
table = soup.table
# find the table rows within the table
table_rows = table.find_all("tr")


table_T = []
for tr in table_rows:
    td = tr.find_all('td')
    row = [i.text for i in td]
    # Only process the cells that have an assigned borough. Ignore cells with a borough that is Not assigned.
    if row != [] and row[1]!= "Not assigned\n":
        # If a cell has a borough but a Not assigned neighborhood, then the neighborhood will be the same as the borough.
        if "Not assigned\n" in row[2]:
            row[2]=row[1]
        table_T.append(row)
# print(table_T)

# define the dataframe columns
column_names = ["PostalCode", "Borough", "Neighborhood"]

# instantiate the dataframe
df = pd.DataFrame(table_T, columns=column_names)
df.head()


# Replace "\n" with " " in columns PostalCode, Borough and Neighborhood

# In[168]:


df["PostalCode"] = df["PostalCode"].str.replace("\n","")
df["Borough"] = df["Borough"].str.replace("\n","")
df["Neighborhood"] = df["Neighborhood"].str.replace("\n","")

df


# In[169]:


df.shape


# Now that you have built a dataframe of the postal code of each neighborhood along with the borough name and neighborhood name, in order to utilize the Foursquare location data, we need to get the latitude and the longitude coordinates of each neighborhood.
# 
# In an older version of this course, we were leveraging the Google Maps Geocoding API to get the latitude and the longitude coordinates of each neighborhood. However, recently Google started charging for their API: http://geoawesomeness.com/developers-up-in-arms-over-google-maps-api-insane-price-hike/, so we will use the Geocoder Python package instead: https://geocoder.readthedocs.io/index.html.
# 
# The problem with this Package is you have to be persistent sometimes in order to get the geographical coordinates of a given postal code. So you can make a call to get the latitude and longitude coordinates of a given postal code and the result would be None, and then make the call again and you would get the coordinates. So, in order to make sure that you get the coordinates for all of our neighborhoods, you can run a while loop for each postal code. 
# 
# Given that this package can be very unreliable, in case you are not able to get the geographical coordinates of the neighborhoods using the Geocoder package, here is a link to a csv file that has the geographical coordinates of each postal code: http://cocl.us/Geospatial_data
# 

# In[170]:


df_geo=pd.read_csv('http://cocl.us/Geospatial_data')
df_geo.head()


# In[171]:


# rename the Postal Code to PostalCode for the geographical coordinates 
df_geo.rename(columns={'Postal Code':'PostalCode'}, inplace=True)
df_geo.head()


# In[172]:


# In this case, we can merge two files by PostalCode 
df_merge = pd.merge(df, df_geo, on='PostalCode' )
df_merge


# ## Explore and cluster the neighborhoods in Toronto. 

# Use geopy library to get the latitude and longitude values of New York City.

# In[173]:


address = 'Toronto, ON'

geolocator = Nominatim(user_agent="toronto_explorer")
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude
print('The geograpical coordinate of Toronoto city are {}, {}.'.format(latitude, longitude))


# Create a map of New York with neighborhoods superimposed on top.

# In[174]:


# create map of New York using latitude and longitude values
map_toronto = folium.Map(location=[latitude, longitude], zoom_start=10)

# add markers to map
for lat, lng, borough, neighborhood in zip(df_merge['Latitude'],df_merge['Longitude'],df_merge['Borough'],df_merge['Neighborhood']):
    label = '{}, {}'.format(neighborhood, borough)
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7,
        parse_html=False).add_to(map_toronto)  
    
map_toronto


# Let's work with only boroughs that contain the word Toronto

# In[175]:


toronto_data = df_merge[df_merge['Borough'].str.contains("Toronto")].reset_index(drop=True)
toronto_data


# In[176]:


# create map of Toronto using latitude and longitude values
new_map_toronto = folium.Map(location=[latitude, longitude], zoom_start=11)

# add markers to map
for lat, lng, label in zip(toronto_data['Latitude'], toronto_data['Longitude'], toronto_data['Neighborhood']):
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7,
        parse_html=False).add_to(new_map_toronto)  
    
new_map_toronto


# #### Define Foursquare Credentials and Version

# In[177]:


CLIENT_ID = '4DBS44JI1BC3QVNDIE1UZPFDFE2OSZJZKJOUJKQUGQKNPQ5M' # your Foursquare ID
CLIENT_SECRET = 'XCF4J5WZ0TFL0OYP1C2RIWLHE4YAPK1WFGAKSPQR01UBGZNO' # your Foursquare Secret
VERSION = '20180605' # Foursquare API version
LIMIT = 100 # A default Foursquare API limit value


# Get the neighborhood's name.

# In[178]:


toronto_data.loc[0, 'Neighborhood']


# In[179]:


neighborhood_latitude = toronto_data.loc[0, 'Latitude'] # neighborhood latitude value
neighborhood_longitude = toronto_data.loc[0, 'Longitude'] # neighborhood longitude value

neighborhood_name = toronto_data.loc[0, 'Neighborhood'] # neighborhood name

print('Latitude and longitude values of {} are {}, {}.'.format(neighborhood_name, 
                                                               neighborhood_latitude, 
                                                               neighborhood_longitude))


# #### Now, let's get the top 100 venues that are in Marble Hill within a radius of 500 meters.

# In[180]:


# type your answer here
LIMIT = 100 # limit of number of venues returned by Foursquare API
radius = 500 # define radius

url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
    CLIENT_ID, 
    CLIENT_SECRET, 
    VERSION, 
    neighborhood_latitude, 
    neighborhood_longitude, 
    radius, 
    LIMIT)



# Send the GET request and examine the resutls

# In[181]:


results = requests.get(url).json()


# In[182]:


# function that extracts the category of the venue
def get_category_type(row):
    try:
        categories_list = row['categories']
    except:
        categories_list = row['venue.categories']
        
    if len(categories_list) == 0:
        return None
    else:
        return categories_list[0]['name']


# In[40]:


venues = results['response']['groups'][0]['items']
    
nearby_venues = json_normalize(venues) # flatten JSON

# filter columns
filtered_columns = ['venue.name', 'venue.categories', 'venue.location.lat', 'venue.location.lng']
nearby_venues =nearby_venues.loc[:, filtered_columns]

# filter the category for each row
nearby_venues['venue.categories'] = nearby_venues.apply(get_category_type, axis=1)

# clean columns
nearby_venues.columns = [col.split(".")[-1] for col in nearby_venues.columns]

nearby_venues.head()


# In[183]:


print('{} venues were returned by Foursquare.'.format(nearby_venues.shape[0]))


# ## 2. Explore Neighborhoods in Toronto

# In[184]:


def getNearbyVenues(names, latitudes, longitudes, radius=500):
    
    venues_list=[]
    for name, lat, lng in zip(names, latitudes, longitudes):
        print(name)
            
        # create the API request URL
        url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            lat, 
            lng, 
            radius, 
            LIMIT)
            
        # make the GET request
        results = requests.get(url).json()["response"]['groups'][0]['items']
        
        # return only relevant information for each nearby venue
        venues_list.append([(
            name, 
            lat, 
            lng, 
            v['venue']['name'], 
            v['venue']['location']['lat'], 
            v['venue']['location']['lng'],  
            v['venue']['categories'][0]['name']) for v in results])

    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
    nearby_venues.columns = ['Neighborhood', 
                  'Neighborhood Latitude', 
                  'Neighborhood Longitude', 
                  'Venue', 
                  'Venue Latitude', 
                  'Venue Longitude', 
                  'Venue Category']
    
    return(nearby_venues)


# In[43]:


# type your answer here
toronto_venues = getNearbyVenues(names=toronto_data['Neighborhood'],
                                   latitudes=toronto_data['Latitude'],
                                   longitudes=toronto_data['Longitude']
                                  )


# In[185]:


print(toronto_venues.shape)
toronto_venues.head()


# Let's check how many venues were returned for each neighborhood

# In[186]:


toronto_venues.groupby('Neighborhood').count()


# #### Let's find out how many unique categories can be curated from all the returned venues

# In[187]:


print('There are {} uniques categories.'.format(len(toronto_venues['Venue Category'].unique())))


# ## 3. Analyze Each Neighborhood

# In[188]:


# one hot encoding
toronto_onehot = pd.get_dummies(toronto_venues[['Venue Category']], prefix="", prefix_sep="")

# add neighborhood column back to dataframe
toronto_onehot['Neighborhood'] = toronto_venues['Neighborhood'] 

# move neighborhood column to the first column
fixed_columns = [toronto_onehot.columns[-1]] + list(toronto_onehot.columns[:-1])
toronto_onehot = toronto_onehot[fixed_columns]

toronto_onehot.head()


# And let's examine the new dataframe size.

# In[189]:


toronto_onehot.shape


# #### Next, let's group rows by neighborhood and by taking the mean of the frequency of occurrence of each category

# In[190]:


toronto_grouped = toronto_onehot.groupby('Neighborhood').mean().reset_index()
toronto_grouped


# #### Let's confirm the new size

# In[191]:


toronto_grouped.shape


# Let's sort the venues in descending order and display the top 10 venues for each neighborhood.

# In[192]:


def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    
    return row_categories_sorted.index.values[0:num_top_venues]


# In[193]:


num_top_venues = 10

indicators = ['st', 'nd', 'rd']

# create columns according to number of top venues
columns = ['Neighborhood']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

# create a new dataframe
neighborhoods_venues_sorted = pd.DataFrame(columns=columns)
neighborhoods_venues_sorted['Neighborhood'] = toronto_grouped['Neighborhood']

for ind in np.arange(toronto_grouped.shape[0]):
    neighborhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(toronto_grouped.iloc[ind, :], num_top_venues)

neighborhoods_venues_sorted.head()


# ## 4. Cluster Neighborhoods

# In[194]:


# set number of clusters
kclusters = 5

toronto_grouped_clustering = toronto_grouped.drop('Neighborhood', 1)

# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(toronto_grouped_clustering)


# In[195]:


# add clustering labels
neighborhoods_venues_sorted.insert(0, 'Cluster Labels', kmeans.labels_)

toronto_merged = toronto_data

# merge manhattan_grouped with manhattan_data to add latitude/longitude for each neighborhood
toronto_merged = toronto_merged.join(neighborhoods_venues_sorted.set_index('Neighborhood'), on='Neighborhood')

toronto_merged.head() # check the last columns!


# In[196]:


toronto_merged = toronto_merged[toronto_merged['Cluster Labels'].notna()]
toronto_merged['Cluster Labels']=toronto_merged['Cluster Labels'].apply(np.int64) 


# visualizing the resulting clusters

# In[200]:


# create map
map_clusters = folium.Map(location=[latitude, longitude], zoom_start=11)

# set color scheme for the clusters
x = np.arange(kclusters)
ys = [i + x + (i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(toronto_merged['Latitude'], toronto_merged['Longitude'], toronto_merged['Neighborhood'], toronto_merged['Cluster Labels']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0.7).add_to(map_clusters)
       
map_clusters


# In[ ]:




