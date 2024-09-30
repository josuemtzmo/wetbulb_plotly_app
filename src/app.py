from dash import Dash, dcc, html
import plotly.graph_objects as go
import pandas as pd
from dash.dependencies import Input, Output
import geopandas as gpd
import json
import numpy as np
import xarray as xr
import os


class MapUtils():
    def __init__(self,datapath='../data'):
        self.path = datapath
        self.load_data()
        self.selected_year = 2024

    def load_data(self):
        # Load population by wetbulb temperature
        data = pd.read_csv(self.path+"/wetbulb_by_country.csv")
        # Set indexes to later select year and polygons.
        self.data = data.set_index(['year','poly_idx']).sort_index()
        

        f = open(self.path+'/coordinates_of_country_polygones.json',)
        self.c_grid_coords_data = json.load(f)

        self.coarsen_population = xr.open_dataset(self.path + '/population_gridded.nc').population.load()
        self.coarsen_wetbulb = xr.open_mfdataset(self.path + "/wetbulb_gridded_data_2*.nc", concat_dim='year',combine='nested').WetBTemp.load()
        
    @staticmethod
    def correct_trend(data, growthrate=0, dt=0):
        return data + dt * growthrate * data

    @staticmethod
    def keep_first(geo):
        if geo.geom_type == 'Polygon':
            point = geo.centroid
        elif geo.geom_type == 'MultiPolygon':
            area = []
            for poly in geo.geoms:
                area.append(poly.area)
            largest_area_idx = np.argmax(area)
            point = geo.geoms[largest_area_idx].centroid
        return {'lon':list(point.coords)[0][0],'lat':list(point.coords)[0][1]}

    def heatmap_by_country(self, index, country, property="population"):
        if property=="wetbulb":
            data = self.coarsen_wetbulb.sel(year = self.selected_year)
        else:
            data = self.coarsen_population

        if country=="Global":
            return data.values.ravel()

        indexes=self.c_grid_coords_data['pix_idxs'][str(index)]
        country_name=self.c_grid_coords_data['name'][str(index)]
        if country != country_name:
            raise ValueError("Mismatch of countries when selecting data.")

        data_raveled = data.values.ravel()
        country_map = data_raveled[indexes]
        X,Y = np.meshgrid(data.lon,data.lat)
        Xr = X.ravel()
        Yr = -Y.ravel()
        X_sel = Xr[indexes]
        Y_sel = Yr[indexes]

        return X_sel, Y_sel, country_map

    def decode_coords(self):
        #Decode coords
        list_coords=[]
        for i, row in self.data.iterrows():
            list_coords.append(json.loads(row.geo_point_2d.replace("\'", "\"")))
        self.data['geo_point_2d'] = list_coords

    
    def country_dict(self):

        # List of countries 
        regions = {"Global":{'label':'Global',
                        'center':{'lat':0,'lon':0},
                        'zoom':1}}

        for i, country in self.data.iterrows():
            if hasattr(country.centroid, "lat"):
                centroid_lat = country.centroid["lat"]
                centroid_lon = country.centroid["lon"]
            else:
                centroid_lat = country.geo_point_2d["lat"]
                centroid_lon = country.geo_point_2d["lon"]

            dict_country = {country['name']: dict( 
                            label="{0}".format(country['name']),
                            center= {"lat":centroid_lat,
                                    "lon":centroid_lon},
                            zoom= country['zoom']
                            ) }
            regions.update(dict_country)

        return regions

    @staticmethod
    def min_range_wb(z):
        # minZ = np.nanmean(z)
        # if minZ < 20:
        return 25
        # else:
            # return minZ


#######################
###### Load data ######
#######################
# Load country polygons
gdf_countries = gpd.read_file('../data/world-administrative-boundaries.geojson',dtype={"geometry": str})


########################
###### Load class ######
########################
MU = MapUtils('../data/')

# Compute centroid of country polygon(s) to display properly in map
gdf_countries['centroid'] = gdf_countries.geometry.apply(lambda _geo: MU.keep_first(_geo))
gdf = gdf_countries.__geo_interface__

MU.decode_coords()

########################
####### APP Dash #######
########################

app = Dash(__name__)
server = app.server
app.title = 'Wet Bulb Temperature'


title =  html.H1(children='Wet Bulb Temperature', id='title')

subtitle = html.P(children='Click over a country to display more information', id='subtitle')

loader = html.Div(children="",id="loader")

regions = MU.country_dict()

dropdown = dcc.Dropdown(
        list(regions.keys()), 
        value="Global",
        id='country-selected', 
        clearable=False,
        optionHeight=30,
        maxHeight=300
    )

slider = dcc.Slider(
        2024,
        2100,
        4,
        id='year-slider',
        value=2024,
        marks=None,
        tooltip={"style": {"color": "white", "fontSize": "20px"},"placement": "bottom", "always_visible": True}
    )

header = html.Div(
            [loader,
            title,
            subtitle,
            dropdown,
            slider],
            id="header"
        )

map = dcc.Graph(id="map", config={"displayModeBar": False}, )

body = [
    header,
    map
]

app.layout = html.Div(children=body)


import plotly.graph_objects as go

layout = go.Layout(
    xaxis=dict(
        domain=[0, 0.6],
        fixedrange=True
    ),
    yaxis=dict(
        domain=[0, 1],
        fixedrange=True
    ),
    xaxis2=dict(
        domain=[0.55, 1],
        anchor="x2",
        fixedrange=True
    ),
    yaxis2=dict(
        # range = [25,35.5],
        domain=[0.52, 0.97],
        anchor="y2",
        fixedrange=True
    ),
    xaxis3=dict(
        domain=[0.55, 1],
        anchor="y3",
        fixedrange=True
    ),
    yaxis3=dict(
        # range = [25,35],
        domain=[0.02, 0.47],
        anchor="x3",
        fixedrange=True
    ),
)

shared_kwards = dict(
                    colorbar=dict(thickness=20, x=0.47, y=0.79,len=0.4, bgcolor="#E0E0E0"),
                    colorscale="Burg",
                    name=""
                    )

density_map = go.Densitymapbox(
    lat=np.arange(0,1,1), 
    lon=np.arange(0,1,1), 
    z=np.zeros(1), 
    hovertemplate = '<br><b>Wet Bulb Temperature </b>: %{z}<br>',
    radius=4,
    **shared_kwards
    )

ignore_global = MU.data.loc[(slice(None), slice("0", "255")), :]

map_conf= go.Choroplethmapbox(
    geojson = gdf,
    locations = ignore_global.index.get_level_values(1).unique(),
    z = ignore_global.loc[2024].WetBTemp_max - 273.15,
    zmin=25, zmax=35,
    text = ignore_global.name,
    hovertemplate = '<b>%{text}</b>'+
                    '<br><b>Max Wet Bulb Temperature </b>: %{z}<br>',
    marker=dict(opacity=0.5),
    **shared_kwards
    )

x=np.arange(0,11,1)
y=np.arange(0,11,1)

shared_kwards_bar_line = dict( hovertemplate = 
                                '<b>Mean Wet Bulb Temperature 25°C:</b> <br>%{y}<br>',
                                name="")

scatter_plot = go.Scatter(
                x=x,y=y, 
                marker=dict(color="darkblue"), 
                xaxis="x2", 
                yaxis="y2",
                showlegend=False,
                **shared_kwards_bar_line
                )

shared_kwards_bar_line = dict( hovertemplate = 
                                '<b>Maximum Wet Bulb Temperature:</b> <br>%{y}<br>',
                                name="")

scatter_plot1 = go.Scatter(
                x=x,y=y, 
                marker=dict(color="crimson"), 
                xaxis="x2", 
                yaxis="y2",
                showlegend=False,
                **shared_kwards_bar_line
                )

shared_kwards_bar_line = dict( hovertemplate = 
                                '<b>Population at wetbulb temperature larger than 30°C:</b> <br>%{y}<br>',
                                name="")

scatter_plot2 = go.Scatter(
                x=x,y=y,
                marker=dict(color="crimson"),
                showlegend=False, 
                xaxis="x3", 
                yaxis="y3",
                **shared_kwards_bar_line)

mapbox=go.layout.Mapbox(
            accesstoken=os.getenv("API_KEY"),
            style="light"
    )

@app.callback(
    Output('country-selected', 'value'),
    Input('map', 'clickData')
    )

def update_map(clickData):
    if clickData is None:
        return "Global"
    else:
        country_clicked = clickData['points'][0]['text']
        if country_clicked in regions.keys():
            return country_clicked
        else: 
            print("Country not found")
            return "Global"
            

@app.callback(Output("map", "figure"),
            Input("country-selected", "value"),
            Input("year-slider", "value"))


def select_bbox(selected_country,year_value):
    country = selected_country
    df_country = MU.data[MU.data.name==country].loc[year_value]
    MU.selected_year = year_value

    if selected_country != "Global":
        index = df_country.index.values.squeeze()
        X,Y,WetBulbT = MU.heatmap_by_country(index, country, property='wetbulb')
        Z = MU.correct_trend(WetBulbT) - 273.15
        density_map.lon = X
        density_map.lat = - Y # Invert y to properly display
        density_map.z = Z
        density_map.text = country
        density_map.zmax = 35
        density_map.zmin = MU.min_range_wb(Z)
        density_map.radius = int(df_country.zoom.values.item()) * 2
        density_map.opacity=1
        copy_df = MU.data.loc[year_value].copy()
        copy_df = copy_df.drop(index.flatten())
        map_conf.locations = copy_df.index
        map_conf.z = MU.correct_trend(copy_df.WetBTemp_max) - 273.15
        map_conf.text = copy_df['name']
        map_conf.marker.opacity = 0.1

    else:
        # Display in global map
        map_conf.locations = ignore_global.index.get_level_values(1).unique()
        map_conf.z = ignore_global.loc[year_value].WetBTemp_max - 273.15
        map_conf.text = ignore_global.name
        map_conf.marker.opacity = 0.5

    # x = np.arange(2024,2101,1)
    C_wetbulb = MU.data[MU.data.name==country].reset_index()

    x = C_wetbulb.year
    y = C_wetbulb.WetBTemp_max - 273.15
    scatter_plot.x = x
    scatter_plot.y = y

    x = C_wetbulb.year
    y = C_wetbulb.WetBTemp_mean - 273.15
    scatter_plot1.x = x
    scatter_plot1.y = y

    
    population_affected = C_wetbulb.population_affected

    scatter_plot2.x = x
    scatter_plot2.y = population_affected

    data = [map_conf, scatter_plot, scatter_plot1, scatter_plot2]

    fig = go.Figure(data=data,layout=layout)

    if selected_country != "Global":
        fig.data[0].update(showscale=False)
        fig.add_trace(density_map)

    regions = MU.country_dict()

    lat = regions[selected_country]["center"]["lat"]
    lon = regions[selected_country]["center"]["lon"]

    
    mapbox.center=go.layout.mapbox.Center(
        lat=lat,
        lon=lon,
    )
    mapbox.pitch=0
    mapbox.zoom=regions[country]["zoom"]
    
    fig.update_layout(
        margin={"r": 0, "t": 0, "l": 0, "b": 0, "pad": 0},
        mapbox=mapbox,
        transition = {'duration': 10, 'easing':'linear'},
        annotations=[
        dict(
            x=0.77,  # X-coordinate for first subplot title
            y=1,    # Y-coordinate (slightly above the plot)
            xref="paper", yref="paper",
            text="{0}'s Wet Bulb Temperature".format(selected_country),  # Title for the first subplot
            showarrow=False,
            font=dict(size=20),
            xanchor="center",
            yanchor="top",
            align = "center",
        ),
        dict(
            x=0.77,  # X-coordinate for second subplot title
            y=0.5,    # Y-coordinate (slightly above the plot)
            xref="paper", yref="paper",
            text="{0}'s Population within wet bulb temperatures larger than 30°C".format(selected_country),  # Title for the second subplot
            showarrow=False,
            font=dict(size=20),
            xanchor="center",
            yanchor="top",
            align = "center",
        )]
    )

    return fig

app.run_server(debug=True)