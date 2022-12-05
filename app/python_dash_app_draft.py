import json

import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px

import os
import sys
sys.path.append("..")
# import pickle
# from python import settings as stg
# from python import extract_shorelines as es
# from python import analyze_shoreline as asl
# from python import correct_tides as ct
# from python import predict as pt
# from python import reconstruct_shoreline as rs
# from python import estimate_pop as ep
from coastsat import SDS_download, SDS_preprocess, SDS_shoreline, SDS_tools, SDS_transects, SDS_slope
# import matplotlib
# import matplotlib.pyplot as plt

import pandas as pd

external_stylesheets = [dbc.themes.BOOTSTRAP]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "1rem 1rem",
}


#%% SIDE BAR
sidebar = html.Div(
    [
        html.H2("Settings", className="display-5"),
        html.Hr(),
        #html.P(
        #    "A simple sidebar layout with navigation links", className="lead"
        #),
        #dbc.Nav(
        #    [
        #        dbc.NavLink("Home", href="/", active="exact"),
        #        dbc.NavLink("Page 1", href="/page-1", active="exact"),
        #        dbc.NavLink("Page 2", href="/page-2", active="exact"),
        #    ],
        #    vertical=True,
        #    pills=True,
        #),
        
        
        html.Label('Polygon of interest'),
        dcc.Input(id="long1",placeholder='Longitude', type="number", style={'width': '50%'}),
        dcc.Input(id="lat1",placeholder='Latitude', type="number", style={'width': '50%'}),
        dcc.Input(id="long2",placeholder='Longitude', type="number", style={'width': '50%'}),
        dcc.Input(id="lat2",placeholder='Latitude', type="number", style={'width': '50%'}),
        dcc.Input(id="long3",placeholder='Longitude', type="number", style={'width': '50%'}),
        dcc.Input(id="lat3",placeholder='Latitude', type="number", style={'width': '50%'}),
        dcc.Input(id="long4",placeholder='Longitude', type="number", style={'width': '50%'}),
        dcc.Input(id="lat4",placeholder='Latitude', type="number", style={'width': '50%'}),
        dcc.Input(id="long5",placeholder='Longitude', type="number", style={'width': '50%'}),
        dcc.Input(id="lat5",placeholder='Latitude', type="number", style={'width': '50%'}),
        html.Hr(),
        
        html.Label('Date range'),
        html.Br(),
        dcc.Input(id="startY",placeholder='YYYY', type="number", style={'width': '40%'}),
        dcc.Input(id="startm",placeholder='MM', type="number", style={'width': '25%'}),
        dcc.Input(id="startd",placeholder='dd', type="number", style={'width': '25%'}),
        html.Br(),
        dcc.Input(id="endY",placeholder='YYYY', type="number", style={'width': '40%'}),
        dcc.Input(id="endm",placeholder='MM', type="number", style={'width': '25%'}),
        dcc.Input(id="endd",placeholder='dd', type="number", style={'width': '25%'}),
        html.Hr(),
        
        html.Label('Satellites'),
        dcc.Checklist(id='satlist',
            options=[
                {'label': 'S2', 'value': 'S2'},
                {'label': 'L7', 'value': 'L7'},
                {'label': 'L8', 'value': 'L8'}
            ],
            value=['S2'],
            labelStyle={'display': 'inline-block','margin-right':'15px'}
        ),
        
        html.Hr(),
        html.Label('Name of the site'),
        dcc.Input(placeholder='Type here',id="sitename",type="text", style={'width': '70%'})
        
        
        
    ],
    style=SIDEBAR_STYLE,
)
#%%

#%% CONTENT BAR
content = html.Div([
    
    html.Div(id="my-output", children=[]),
    html.Br(),
    
    dcc.Graph(id='my-graph', figure={})
    
],         
style=CONTENT_STYLE)
#%%
        
app.layout = html.Div([dcc.Location(id="url"), sidebar, content])

#%% CALLBACK

@app.callback(
    Output(component_id='my-output', component_property='children'),
    Input('long1', 'value'),
    Input('long2', 'value'),
    Input('long3', 'value'),
    Input('long4', 'value'),
    Input('long5', 'value'),
    Input('lat1', 'value'),
    Input('lat2', 'value'),
    Input('lat3', 'value'),
    Input('lat4', 'value'),
    Input('lat5', 'value'),
    Input('startY', 'value'),
    Input('startm', 'value'),
    Input('startd', 'value'),
    Input('endY', 'value'),
    Input('endm', 'value'),
    Input('endd', 'value'),
    Input('satlist', 'value'),
    Input('sitename', 'value'))
def create_inputs(long1,long2,long3,long4,long5,lat1,lat2,lat3,lat4,lat5,
                  startY,startm,startd,endY,endm,endd,satlist,satname):
    # region of interest (longitude, latitude)
    polygon = [[[long1, lat1],  
                [long2, lat2], 
                [long3, lat3],
                [long4, lat4],
                [long5, lat5]]] 
    # it's recommended to convert the polygon to the smallest rectangle (sides parallel to coordinate axes)       
    polygon = SDS_tools.smallest_rectangle(polygon)
    # date range
    dates = [str(startY)+'-'+str(startm)+'-'+str(startd),str(endY)+'-'+str(endm)+'-'+str(endd)]
    # satellite missions
    sat_list = satlist
    # name of the site
    sitename = satname
    # directory where the data will be stored
    filepath = os.path.join(os.getcwd(), 'data')
    # put all the inputs into a dictionary
    inputs = {'polygon': polygon, 'dates': dates, 'sat_list': sat_list, 'sitename': sitename, 'filepath':filepath}
    json.dumps(inputs)
    
    return 

#%%    

if __name__ == '__main__':
    app.run_server(debug=True)