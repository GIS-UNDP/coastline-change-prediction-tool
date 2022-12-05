# Creating Coastal Prediction Apps using Google Earth Engine

## Useful links
* [Template script](https://github.com/Space4Dev/coastline-predictor/blob/main/app/template_script.js)
* [Google Earth Engine website](https://earthengine.google.com/)
* [Google Earth Engine API](https://code.earthengine.google.com/)
* [Google Earth Engine Documentation](https://developers.google.com/earth-engine)
* [Gradient color palette generator](https://coolors.co/gradient-palette)
* [Information about CSS HEX codes](https://www.w3schools.com/css/css_colors_hex.asp)

## App Example
You can see below an example of a Google Earth Engine app that has been created to display predictions of the coast erosion in Fongafale, Tuvalu. The app is composed of a right panel with the UNDP logo, its title and a brief summary, a right panel with checkboxes and a legend to display your shorelines. And finally it features a checkbox on the bottom right corner to display a population density layer.  
At the end of this Readme, you will be able to create a similar GEE app focused on your area of interest.
![img](img/app_example.png)

# I. Google Earth Engine Presentation

## A.	Signing up for GEE

To create a GEE account, you will need a Google account.

![Google Earth Engine website : https://earthengine.google.com](img/gee_screen.png)

* Go to https://earthengine.google.com and click on **Sign up**.
* Select the Google account you want to use and sign in.
* Fill in the form with the requested information and click on **Submit**.
* Wait for the acceptation e-mail.

Google Earth Engine signing up is not automatic and may take a few minutes or even be denied if you filled in wrong information. Once your GEE access has been granted you’ll have access to the Application Programming Interface at https://code.earthengine.google.com.

## B. Google Earth Engine's API

![title](img/gee_api.png)

***0 : Search Bar***  
With this search bar, you can look up and import to your project publicly available satellite datasets.  
  
***1.1 : Scripts***  
On this tab you can see all the repositories you have access to, either by being the owner or have a writer/reader access. Inside those repositories are the scripts. You can also consult example scripts the Examples folder.  
  
***1.2 : Docs***  
The GEE API uses the Javascript coding language with additional functions and methods that are specific to GIS. You can find some brief documentation in this tab. You can find detailed examples at https://developers.google.com/earth-engine.  
  
***1.3 : Assets***  
You will manage from this tab all the external ressources you will need to import for your projects. They can be images, shapefiles, layers…  
  
***2 : Coding panel***  
You can write code in this panel. It features 5 buttons. ‘Get Link’ will generate a link for you to share your script with someone else. ‘Save’ will save your script in the repository. ‘Run’ will run your code within the API. ‘Reset’ will clear your script. From ‘Apps’ you will be able to create and update apps containing your code.  
  
***3.1 : Inspector***  
The inspector mode allows you click anywhere on the map and display coordinates and detailed information in this tab.  
  
***3.2 : Console***  
You might want to resolve some errors in your code by displaying variables or éléments from your script. Everything you print by using the print() function will be displayed here.  
  
***3.3 : Tasks***  
On this tab, you will be able to see the evolution of your external imports.  
  
***4 : Map***  
On this panel, you will see the preview of your future application when clicking on ‘Run’ from the coding panel.


## C. Setting up your project

In order to create your app you need to create a repository and a script. You don’t need to create a repository if you already have one that you want to use for this project.  

***Creating a new repository :***  
On the top left panel, in the *Script* tab, select **NEW** > **Repository** and give it the name you want to.

![img](img/new_repository.PNG)

  
***Creating a script :***  
On the top left panel, in the *Script* tab, select **NEW** > **File**, make sure it will be created in the repository
you want to use and name it.

![img](img/new_script.PNG)

Now that you have your blank new script, you may find a **template script** to make coastal erosion apps [here](https://github.com/Space4Dev/coastline-predictor/blob/main/app/template_script.js).  
Copy its content and paste it into your script.

# II. Creating a GEE App

## A. Creating the geometries

To start off, you need to create two geometries on the map panel.  
The first one is called ``geometry_center`` and is used the center the application to your area of interest when you run it. As you can see, the map is centered to the United States by default.

![img](img/draw_geometry.png)

Click on **Draw a geometry** as you can see on the screenshot above. Then just go to your area of interest and place your points so you can have your geometry. This geometry doesn't need to be very precise or to perfectly fit the contours of the area as it will not be visible.  
On the top right corner of the map panel you can switch to the satellite view.

![gif](img/geo_gif.gif)

Repeating the steps in the above GIF you will be able to create your first geometry : ``geometry_center``.  
Now repeat the process and create a new geometry by clicking on **new layer**, below your created geometry. This second geometry needs to be named ``geometry_pop`` and will be used to clip the world's population layer to only your area of interest. This second one will need to be very precise and follow as much as possible the contours the area you want to cover.  

## B. Importing your shorelines

To import your shorelines to your GEE app, you will need to have them in the shapefile format.  
A shapefile is composed of 5 different files : *.cpg, .dbf, .prj, .shp, .shx*, you can import the five of them directly into Google Earth Engine or have them zipped together in a *.zip*.  
  
![shapefileimg](img/shapefile.PNG)



In the top left panel, click on the **Assets** tab, then select **NEW** > **Shape files (.shp, .shx, .dbf, .prj or .zip)** and **SELECT** your shapefile or juste drag and drop it. You can now choose under which name you wish to save it in your assets.  
If you're working on several apps, you might want to create a dedicated folder and change the path by adding ``name_of_your_folder/`` just before your file's name.  
Scroll to the end of the window and click on **UPLOAD**. Your import will appear on the top right panel under the **Tasks** tab. After one or two minutes, your import will appear blue if successful. If one file is missing, the import will fail and appear red.

![img](img/importing.PNG)

You can find your import under the **Assets** tab, right where you decided to store it. Now, repeat the operation for each shapefile you have (for every year or predicted year you want to cover with your app).

![gif](img/importing_gif.gif)

Once you have uploaded your shapefiles, you need to import them into your script, as the GIF above.  
As those shapefiles will be used in a publicly available app, click on the sharing options of the shapefile and tick **Anyone can read**. Now, click on the arrow next to the sharing options button and your shapefile will appear on a band, on top of your script, alongside the geometries you previously drew.  
By default, your files will be imported as *table*, *table2*, etc, the last step is to change this name to **output_linesYYYY** with YYYY representing the year this shoreline represents.

## C. Adapting the template script

### Variables setup
Now, you should have your script where the content of [template_script.js](https://github.com/Space4Dev/coastline-predictor/blob/main/app/template_script.js) had been pasted, your two geometries (``geometry_center`` and ``geometry_pop``) and your imported shorelines (one for each year you want to represent on your app).  
For the app to be operational, you need to oper some changes in the **INPUT SETTINGS** section of the code where you will find the following variables :
<br>
<br>
```javascript
var header_text = 'Place, Country : Coastal Changes Prediction' ;
var description = 'This tool is used to predict the future coastal changes along the built-up areas in Place, Country using sets of satellite images and time-series modelling.' ;
```
Those two variables represent the header and the description of the app's right panel. For this one, just replace Place and Country. You can change the text as you like, as long as you respect the syntax above.
<br>
<br>
```javascript
var actual_tab = ['2015','2016','2017','2018','2019','2020','2021'] ;
var predict_tab = ['2022','2023','2024'] ;
```
The yearly shorelines you will display on your app divide into two categories : the actual shorelines that comes from real satellite images, and the predicted shorelines which were predicted by the algorithm using the actual shorelines. Modify the ``actual_tab`` and ``predict_tab`` arrays so that they contain the actual and predicted years you want to feature in your app.
<br>
<br>
```javascript
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
```
Here, do the same for the ``SL_tab`` variable, it must contain every year you want to display on your app and following this syntax for the names : *output_linesYYYY*.
<br>
<br>
```javascript
var surface2022_low = 2.243701739210003 ;
var surface2022_high = 4.67197959152358 ;
var surface2023_low = 2.7847990351532235 ;
var surface2023_high = 7.285063151571775 ;
var surface2024_low = 5.2224751063562795 ;
var surface2024_high = 13.607053564567735 ;

var surface_tab = [surface2022_low, surface2022_high,
                  surface2023_low, surface2023_high,
                  surface2024_low, surface2024_high] ;
```
The algorithm is able to give you the estimated lost surface between your last actual shoreline and any of your predicted shorelines so that we can compute a value of the estimated threatened population.  
Replace the given values with the ones the algorithm gave you. Don't forget that if you have more predictions you can create new variables and insert them in ``surface_tab`` following the same syntax. If you don't have surfaces values for your predictions, leave the values as they are but you won't be able to exploit the estimated threatened population results.
<br>
<br>

### Palette setup
```javascript
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
```
The ``palette`` variable is an array of colors that must contain as many elements as the total number of yearly shorelines you wish to display (actual and predicted combined).  
Here, colors are represented in CSS HEX. Each color is a composite of a red, a green and a blue value going from 0 to 255. In hexadecimal, they go from *00* to *FF*. For more information about CSS HEX codes, you can refer to this [link](https://www.w3schools.com/css/css_colors_hex.asp).


This [website](https://coolors.co/gradient-palette) allows you to create custom gradient color palettes and export them as an array that you can paste in your script. The below GIF shows you how to process to create a gradient of 8 colors going from blue to red. Of course, you can choose to include any color palette you want.

![gif](img/gradient.gif)
<br>
<br>
### Checkboxes setup
The last step to adapt the template script is to setup the checkboxes in the **SHORELINES LAYERS MANAGEMENT** section of the script. 

```javascript
// Actual shorelines checkbox function
CheckTab[INDEX].onChange(function(checked) {
  layer_tab[INDEX].setShown(checked);
}) ;
```
For each actual shoreline you want to display in your app, adapt this part of script by ajusting the **INDEX** value. You must have as many Check function as you have actual shorelines following the syntax above.  
If you have 6 actual shorelines, you must have 6 functions with indexes going from 0 to 5.  


Now for the predicted shorelines checkboxes, the function to use is slightly different :
```javascript
// Predicted shorelines checkbox function
CheckTab[INDEX].onChange(function(checked) {
  layer_tab[INDEX].setShown(checked) ;
  chart_cond = actualize_cc() ;
  chart_panel.style().set('shown',chart_cond) ;
}) ;
```
Once again, your script needs to feature the exact same number of functions as you the number of predicted shorelines you have. You just need to modify the indexes but starting from the index following the last actual shorelines checkbox function you created.  


Example :  
If you have 6 actual shorelines and 3 predicted shorelines, create 6 actual shorelines functions with indexes from 0 to 5 and 3 predicted shorelines functions with indexes from 6 to 8.

## D. Publishing your app

You can now press the **Run** button on top of the Code Editing panel. If no error appears on the **Console** tab, then press the **Save** button and your app is ready to be published.
Press the **Apps** button of the Code Editing panel and select **NEW APP**.
![img](img/publish.PNG)
Here, you can give your app a name, manage the access, add a thumbnail and a description. On the *Source code* section, you can choose what script your app will be running. You can either choose **Current contents of editor** with your script open or **Repository script path** and select the path to your script. Once you are done, click on **PUBLISH**.
![img](img/app_galery.PNG)
Your freshly created app now appears on this App managing window where you can see all of your previously created apps. Click on the name of your app to launch it. You may have to wait a few minutes after its creation for the app to be accessible.  
As you modify your script, you can also update your app by click on its link in the app management window.
