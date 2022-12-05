import ipywidgets as widgets
from IPython.display import display

def get_inputs():
    
    print('Polygon of interest')
    lat1 = widgets.FloatText(
        value=0,
        description='Latitude 1:',
        disabled=False
    )
    long1 = widgets.FloatText(
        value=0,
        description='Longitude 1:',
        disabled=False
    )
    display(lat1)
    display(long1)
    print('---------------------------------------------')
    lat2 = widgets.FloatText(
        value=0,
        description='Latitude 2:',
        disabled=False
    )
    long2 = widgets.FloatText(
        value=0,
        description='Longitude 2:',
        disabled=False
    )
    display(lat2)
    display(long2)
    print('---------------------------------------------')
    lat3 = widgets.FloatText(
        value=0,
        description='Latitude 3:',
        disabled=False
    )
    long3 = widgets.FloatText(
        value=0,
        description='Longitude 3:',
        disabled=False
    )
    display(lat3)
    display(long3)
    print('---------------------------------------------')
    lat4 = widgets.FloatText(
        value=0,
        description='Latitude 4:',
        disabled=False
    )
    long4 = widgets.FloatText(
        value=0,
        description='Longitude 4:',
        disabled=False
    )
    display(lat4)
    display(long4)
    print('---------------------------------------------')
    lat5 = widgets.FloatText(
        value=0,
        description='Latitude 5:',
        disabled=False
    )
    long5 = widgets.FloatText(
        value=0,
        description='Longitude 5:',
        disabled=False
    )
    display(lat5)
    display(long5)
    print('=============================================')
    start = widgets.DatePicker(
        description='Start date',
        disabled=False
    )
    end = widgets.DatePicker(
        description='End date',
        disabled=False
    )
    display(start)
    display(end)
    print('=============================================')
    print('Satellite missions')
    S2 = widgets.Checkbox(
        value=True,
        description='Sentinel-2',
        disabled=False,
        indent=False
    )
    L7 = widgets.Checkbox(
        value=False,
        description='Landsat-7',
        disabled=False,
        indent=False
    )
    L8 = widgets.Checkbox(
        value=False,
        description='Landsat-8',
        disabled=False,
        indent=False
    )
    display(S2)
    display(L7)
    display(L8)
    print('=============================================')
    print('Name of the site')
    sitename = widgets.Text(
        value='',
        placeholder='Type here',
        description='Site name:',
        disabled=False
    )
    display(sitename)
    
    return lat1,long1,lat2,long2,lat3,long3,lat4,long4,lat5,long5,start,end,S2,L7,L8,sitename