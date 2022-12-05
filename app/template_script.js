//////////////////////////////////// GEOMETRY CREATION ///////////////////////////////////////////////////////////

/* 

You need to create 2 geometries :
- geometry_center : a geometry on which the app will be centered when you run it (your area of interest)
- geometry_pop : a geometry that will represent the area where the population lives, this one needs to be
  as precise as possible.
  
*/

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



/////////////////////////////////// INPUT SETTINGS /////////////////////////////////////////////////////////////////

//Header and description
// Enter here the text you want to be displayed as a title and description for your app on the right panel
var header_text = 'Place, Country : Coastal Changes Prediction' ;
var description = 'This tool is used to predict the future coastal changes along the built-up areas in Place, Country using sets of satellite images and time-series modelling.'

//Selected years
// Fill the tables depending on the actual and predicted years you want to display in the app
var actual_tab = ['2015','2016','2017','2018','2019','2020','2021'] ;
var predict_tab = ['2022','2023','2024'] ;

// Shorelines Tab
// Enter in the tab the corresponding shorelines as imported in your Assets
var SL_tab = [output_lines2015,
              output_lines2016,
              output_lines2017,
              output_lines2018,
              output_lines2019,
              output_lines2020,
              output_lines2021,
              output_lines2022,
              output_lines2023,
              output_lines2024] ;


// Enter here the surfaces of erosion obtained through the algorithm
var surface2022_low = 2.243701739210003 ;
var surface2022_high = 4.67197959152358 ;
var surface2023_low = 2.7847990351532235 ;
var surface2023_high = 7.285063151571775 ;
var surface2024_low = 5.2224751063562795 ;
var surface2024_high = 13.607053564567735 ;

var surface_tab = [surface2022_low, surface2022_high,
                  surface2023_low, surface2023_high,
                  surface2024_low, surface2024_high] ;
                  
                  
//Create a palette using the JET colormap
// This palette needs to have the exact same number of colors than the number of shorelines (actual+predicted)
// you want to display.
// More info about RGB and CSS HEX codes : https://www.w3schools.com/css/css_colors_hex.asp
// How to create a gradient : https://coolors.co/gradient-palette
var palette = [
    '00BFFF',
    '00FFFF',
    '40FFBF',
    '80FF80',
    'BFFF40',
    'FFFF00',
    'FFBF00',
    'FF8000',
    'FF4000',
    'FF0000'
];

// Before launching the app, you need to adapt a few checkboxes settings in the SHORELINE LAYERS MANAGEMENT section.

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


/////////////////////////////////// MAP SETTINGS /////////////////////////////////////////////////////////////////////////
//Set up a satellite background
Map.setOptions('Satellite');

// Center the map on West Point
Map.centerObject(geometry_center, 14);

//Change style of cursor to 'crosshair'
Map.style().set('cursor', 'crosshair');
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


////////////////////////////////// FUNCTIONS /////////////////////////////////////////////////////////////////////////////
/*
    * Used to create and style 1 row of the legend including the corresponding checkbox.
    * @param check - the corresponding ui.Checkbox
    * @param color - a css-style color code (ex: "FF0080")
    * @return ui.Panel - the created panel
*/
var makeRow = function(color, check) {
  
      // Create the label that is actually the colored box.
      var colorBox = ui.Label({
        style: {
          backgroundColor: color,
          // Use padding to give the box height and width.
          height: '4px',
          width: '30px',
          margin: '14px 0px 0px 5px'
        }
      });
      return ui.Panel({
        widgets: [check,colorBox],
        layout: ui.Panel.Layout.Flow('horizontal')
      });
} ;


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



/////////////////////////////////// POPULATION SETTINGS ///////////////////////////////////////////////////////////////////

// Layer used for population density
var datapop = ee.ImageCollection("WorldPop/GP/100m/pop").mosaic().clip(geometry_pop) ;

// Average, max and minimum population density for the selected area
var density = datapop.reduce(ee.Reducer.first()).reduceRegion(ee.Reducer.mean(),geometry_pop,100).get('first').getInfo();
var min_dens = datapop.reduce(ee.Reducer.first()).reduceRegion(ee.Reducer.minMax(),geometry_pop,100).get('first_min').getInfo() ;
var max_dens = datapop.reduce(ee.Reducer.first()).reduceRegion(ee.Reducer.minMax(),geometry_pop,100).get('first_max').getInfo() ;

// The gradient palette that is used to represent the population density
var palette_pop = {min: min_dens , max : max_dens,palette : ['#FFFFFF','#FFFFDB','#FFFFB6','#FFFF92','#FFFF6D',
                                                            '#FFFF49','#FFFF24','#FFFF00','#FFE300','#FFC600',
                                                            '#FFAA00','#FF8E00','#FF7100','#FF5500','#FF3900',
                                                            '#FF1C00','#FF0000','#E30000','#C60000','#AA0000',
                                                            '#8E0000','#710000','#550000','#390000','#1C0000']};

//Number of actual and predicted shorelines
var nb_actual = ee.List(actual_tab).length().getInfo() ;
var nb_predict = ee.List(predict_tab).length().getInfo() ;

// Estimated threatened population per year
var list_dict = [] ;
for (var i = 0 ; i<nb_predict ; i++) {
  var pop_low = Math.round(density * surface_tab[2*i]) ;
  var pop_high = Math.round(density * surface_tab[2*i + 1]) ;
  list_dict.push(predict_tab[i],ee.List([pop_low,pop_high])) ;
}
var dict = ee.Dictionary(list_dict) ;

var chart = ui.Chart.array.values(dict.toArray(), 0, predict_tab)
.setOptions({
  hAxis: {
      title: 'Predicted years'
    },
    legend: {position: 'none'},
    pointSize: 0,                
    lineSize: 3,
    series: {
    0: {color: '00FF00', pointSize: 5},
    1: {color: 'FF0000', pointSize: 5}
  },
    chartArea: {backgroundColor: 'rgba(0,0,0,0)', color: 'rgba(0,0,0,0)'},
    style : {backgroundColor : 'rgba(0,0,0,0)', color : 'rgba(0,0,0,0)'}
});

var pop_title = ui.Label({
    value:'Estimated threatened population',
    style: {fontSize: '16px', fontWeight: 'bold', color:'white', backgroundColor:'rgba(80,80,80,0)'}
  });

var chart_panel = ui.Panel({
  widgets : [pop_title,chart],
  style : {
    position : 'bottom-left',
    backgroundColor : 'rgba(80,80,80,0.70)',
    width : '350px',
    shown : false
  }
}) ;

Map.add(chart_panel)

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////



///////////////////////////////// INTERFACE SETTINGS /////////////////////////////////////////////////////////////
//App title
var header = ui.Label(header_text, {fontSize: '24px', fontWeight: 'bold', color: '#333333', stretch:'both'});

//UNDP Logo (thumbnail)
var logo = ee.Image('users/augustingalloo/UNDP-logo-350x293_georef').visualize({
    bands:  ['b1', 'b2', 'b3'],
    min: 0,
    max: 255
    });
var thumb_logo = ui.Thumbnail({
    image: logo,
    params: {
        dimensions: '73x146',
        format: 'png'
        },
    style: {padding :'0'}
});

var header_logo = ui.Panel({widgets:[thumb_logo,header],layout:ui.Panel.Layout.Flow('horizontal')});

//App summary
var text = ui.Label(description,{fontSize: '15px'});
    
// Creation of the right panel with logo and presentation
var right_panel = ui.Panel({
  widgets:[header_logo, text],//Adds logo, header and text
  style:{width: '300px',position:'middle-right'}});

// Instructions for the left panel 
var instruc = ui.Label({
    value:'Select shorelines to display',
    style: {fontSize: '16px', fontWeight: 'bold'}
  });
  
// Creation of the left panel
var left_panel = ui.Panel({
  widgets:[instruc],
  style:{position:'middle-left'}});


// Displaying panels
Map.add(right_panel);
ui.root.insert(0,left_panel);

// Checkboxes Label (left panel)
var checkbox_label1 = ui.Label({value:'Shorelines per year',
style: {fontWeight: 'bold', fontSize: '15px', margin: '10px 5px'}
});
var checkbox_label2 = ui.Label({value:'Predicted Shorelines',
style: {fontWeight: 'bold', fontSize: '15px', margin: '10px 5px'}
});

//Checkboxes creation and Tab (left panel)
var CheckTab = [] ;
var years_tab = actual_tab.concat(predict_tab) ;
for (var i=0 ; i<nb_actual + nb_predict; i++) {
  CheckTab.push(ui.Checkbox(years_tab[i]).setValue(false)) ;
}

// Creating the legends
var SL_legend_actual = ui.Panel({
  style: {
    position: 'bottom-left'
  }
  
});
var SL_legend_predicted = ui.Panel({
  style: {
    position: 'bottom-left'
  }
  
});
           
// Add color, checkboxes and names to the legends
// 2015-2021 shorelines legend
for (var i = 0; i < nb_actual; i++) {
  SL_legend_actual.add(makeRow(palette[i],CheckTab[i]));
  } 
// Predicted shorelines legend
// 2022-2024
for (var j = nb_actual; j < nb_predict + nb_actual; j++) {
  SL_legend_predicted.add(makeRow(palette[j],CheckTab[j]));
  } 

//Adding elements to the left panel
left_panel.add(checkbox_label1);
left_panel.add(SL_legend_actual);
left_panel.add(checkbox_label2);
left_panel.add(SL_legend_predicted);
//////////////////////////////////////////////////////////////////////////////////////////////////////////




/////////////////////////////  SHORELINES LAYERS MANAGEMENT ///////////////////////////////////////////////

//Creating the layers for each shoreline
// and adding each created layer to the map ;
var layer_tab = [] ;
for (var i = 0 ; i < nb_actual + nb_predict ; i ++) {
  layer_tab.push(ui.Map.Layer(SL_tab[i], {color: palette[i]}, 'colored', false)) ;
  Map.add(layer_tab[i]) ;
}

// Creation of the condition for the chart to appear
var chart_cond ;
var actualize_cc = function () {
  var res = layer_tab[nb_actual].getShown() ;
  for (var i = 0 ; i < nb_predict - 1; i++) {
    res = res || layer_tab[nb_actual+i+1].getShown() ;
  }
  return res ;
} ;


// /!\ MODIFICATION NEEDED HERE
// Actual shorelines checkbox action management
// You need to create a new onChange function for every shoreline in actual_tab, incrementing the indexes
// If you have 8 actual shorelines, the indexes should go from 0 to 7.


CheckTab[0].onChange(function(checked) {
  layer_tab[0].setShown(checked);
}) ;

CheckTab[1].onChange(function(checked) {
  layer_tab[1].setShown(checked);
}) ;

CheckTab[2].onChange(function(checked) {
  layer_tab[2].setShown(checked);
}) ;

CheckTab[3].onChange(function(checked) {
  layer_tab[3].setShown(checked);
}) ;

CheckTab[4].onChange(function(checked) {
  layer_tab[4].setShown(checked);
}) ;

CheckTab[5].onChange(function(checked) {
  layer_tab[5].setShown(checked);
}) ;

CheckTab[6].onChange(function(checked) {
  layer_tab[6].setShown(checked);
}) ;


// /!\ MODIFICATION NEEDED HERE
// Predicted shorelines checkbox action management
// You need to create a new onChange function for every shoreline in predict_tab, incrementing the indexes and starting
// from the lasts one you created just above

CheckTab[7].onChange(function(checked) {
  layer_tab[7].setShown(checked) ;
  chart_cond = actualize_cc() ;
  chart_panel.style().set('shown',chart_cond) ;
}) ;

CheckTab[8].onChange(function(checked) {
  layer_tab[8].setShown(checked) ;
  chart_cond = actualize_cc() ;
  chart_panel.style().set('shown',chart_cond) ;
}) ;

CheckTab[9].onChange(function(checked) {
  layer_tab[9].setShown(checked) ;
  chart_cond = actualize_cc() ;
  chart_panel.style().set('shown',chart_cond) ;
}) ;


/////////////////////////////////////////////////////////////////////////////////////////////////////



///////////////////////////// INSPECTOR /////////////////////////////////////////////////////////////

// Creating the inspector panel
var inspector = ui.Panel({
  layout: ui.Panel.Layout.flow('horizontal'),
  style: {backgroundColor:'rgba(80,80,80,0.70)'}
});

inspector.add(ui.Label({value:'Click to get population density',style:{color:'white',backgroundColor:'rgba(0,0,0,0)', fontWeight:'bold'}}));

Map.add(inspector);

// Function to get the density by clicking on the map
Map.onClick(function(coords){
  
  inspector.clear();
  inspector.style().set('shown', true);
  inspector.add(ui.Label('Loading...', {color: 'white',backgroundColor:'rgba(0,0,0,0)'}));
  
  var point = ee.Geometry.Point(coords.lon, coords.lat);
  var reduce = datapop.reduce(ee.Reducer.first());
  var sampledPoint = reduce.reduceRegion(ee.Reducer.first(), point, 30);
  var computedValue = sampledPoint.get('first');  

// Request the value from the server and use the results in a function.
  computedValue.evaluate(function(result) {
  inspector.clear();

// Add a label with the results from the server.
  inspector.add(ui.Label({
      value: 'density: ' + (100*result).toFixed(2) + ' inhab./km²',
      style: {stretch: 'vertical',color:'white',backgroundColor:'rgba(0,0,0,0)'}
    }));

// Add a button to hide the Panel.
    inspector.add(ui.Button({
      label: 'X',
      style:{backgroundColor:'rgba(0,0,0,0)'},
      onClick: function() {
        inspector.style().set('shown', false);
      }
    }));
  });
});
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////




///////////////////////////// POPULATION LAYER MANAGEMENT /////////////////////////////////////////////////////////////////////////////

// Visualization settings
var visualization = {
  bands: ['population'],
  min: min_dens,
  max: max_dens,
  palette: palette_pop.palette
};


// Population layer creation
var pop_layer = ui.Map.Layer(datapop, visualization, 'Population',false,0.5) ;
Map.add(pop_layer) ;

//Checkbox management
var pop_check = ui.Checkbox({label: 'Population density layer',value:false,style:{color:'white',backgroundColor:'rgba(0,0,0,0)'}});
var source = ui.Label({value:'source : worldpop.org',style:{fontSize:'11px',fontWeight:'italic',color:'white',backgroundColor:'rgba(0,0,0,0)'},targetUrl:'https://www.worldpop.org/'});

// Gradient legend
var lon = ee.Image.pixelLonLat().select('latitude');
var gradient = lon.multiply((palette_pop.max-palette_pop.min)/100.0).add(palette_pop.min);
var legendImage = gradient.visualize(palette_pop);

var legendTitle = ui.Label({
value: 'Population density (inhab./km²)',
style: {
fontWeight: 'bold',
fontSize: '18px',
margin: '0 0 4px 0',
padding: '0',
color:'white',
backgroundColor:'rgba(80,80,80,0)',
}});

// create thumbnail from the image
var thumbnail = ui.Thumbnail({
image: legendImage,
params: {bbox:'0,0,10,100', dimensions:'10x200'},
style: {padding: '1px', position: 'bottom-center', backgroundColor:'rgba(80,80,80,0)'}
});

var min_text = ui.Label({
  value : Math.round(100*min_dens),
  style : {
    color : 'white',
    backgroundColor : 'rgba(80,80,80,0)'
  }
}) ;

var max_text = ui.Label({
  value : Math.round(100*max_dens),
  style : {
    color : 'white',
    backgroundColor : 'rgba(80,80,80,0)'
  }
}) ;

var txt_panel = ui.Panel({
  widgets : [max_text,ui.Label({style: {stretch: 'vertical'}}),min_text],
  style : {
    backgroundColor : 'rgba(80,80,80,0)',
    stretch: 'vertical'
  }
});

var aux_panel = ui.Panel({
  widgets : [thumbnail,txt_panel],
  layout : ui.Panel.Layout.Flow('horizontal'),
  style : {
    backgroundColor : 'rgba(80,80,80,0)'
  }
}) ;

var gradient_panel = ui.Panel({
  widgets : [legendTitle, aux_panel],
  style : {
  width : '160px',
  shown : false,
  backgroundColor : 'rgba(80,80,80,0)'
  }
}) ;

var pop_layer_panel = ui.Panel({widgets:[gradient_panel,pop_check, source],style:{position:'bottom-right',backgroundColor:'rgba(80,80,80,0.70)'}});

Map.add(pop_layer_panel);

var docheckboxpop = function() {
pop_check.onChange(function(checked){
pop_layer.setShown(checked) ;
right_panel.style().set('shown',!checked) ;
gradient_panel.style().set('shown',checked) ;
})} ;
docheckboxpop();
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////