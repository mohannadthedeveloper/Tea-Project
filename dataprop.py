#Satellite Data Preparation and Processing
import pandas as pd
import geopandas as gpd
import fiona
import ee

#Init GE project
start_date = '2015-01-01'
end_date = '2026-10-01'
ee.Authenticate()
try:
    ee.Initialize(project='tea-511003') 
    print("Local Earth Engine connection fully authorized.")
except Exception as e:
    print(f"Connection error: {e}")

#init KML reader
fiona.drvsupport.supported_drivers['KML'] = 'rw'
gdf = gpd.read_file('tea.kml')
polygon_shape = gdf.geometry.iloc[0]
raw_coords = list(polygon_shape.exterior.coords) # Format: [(lon1, lat1), (lon2, lat2), ...]
clean_coords = [[coord[0], coord[1]] for coord in raw_coords] #flatten the coords

#define our Region of Interest and getting our images from the Sentinel 2 satellite (Changoi Farm)
landsat = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterDate(start_date, end_date)
roi = ee.Geometry.Polygon([clean_coords])
landsat_aoi = landsat.filterBounds(roi)

def compute_metrics(img):
    #Account for cloud cover
    qa = img.select('QA60')
    cloud_bit = 1 << 10
    cirrus_bit = 1 << 11
    clear_sky_mask = qa.bitwiseAnd(cloud_bit).eq(0).And(qa.bitwiseAnd(cirrus_bit).eq(0))
    img_masked = img.updateMask(clear_sky_mask)
    #Compute NDVI and NDWI of image data
    ndvi = img_masked.normalizedDifference(['B8', 'B4']).rename('NDVI')
    ndwi = img_masked.normalizedDifference(['B8', 'B11']).rename('NDWI')
    return img_masked.addBands(ndvi).addBands(ndwi)

#define our Region of Interest and getting our images from the Sentinel 2 satellite (Changoi Farm)
roi = ee.Geometry.Polygon([clean_coords])
landsat = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterDate(start_date, end_date).filterBounds(roi).map(compute_metrics))
landsat_aoi = landsat.filterBounds(roi)

#Pulling average NDVI and NDWI for each image
def reduce_region(img):
    mean_ndvi = img.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=roi,
        scale=30
    ).get('NDVI')
    mean_ndwi = img.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=roi,
            scale=30
        ).get('NDWI')
    date = ee.Date(img.get('system:time_start')).format('YYYY-MM-dd')
    return ee.Feature(None, {'date': date, 'ndvi': mean_ndvi, 'ndwi': mean_ndwi})

ndvi_data = landsat_aoi.map(reduce_region)
raw_features = ndvi_data.getInfo()
records = [f['properties'] for f in raw_features['features']]
ndvi = pd.DataFrame(records)
#reformat date
ndvi['date'] = pd.to_datetime(ndvi['date'], errors='coerce')
ndvi.set_index('date', inplace=True)
monthly_ndvi = ndvi.resample('ME').mean()
monthly_ndvi.index = monthly_ndvi.index + pd.Timedelta(days=1) #prevents chronological misalignment
monthly_ndvi = monthly_ndvi.ffill().dropna() #resolves missing values

#Now time to merge the Satellite data with the tea prices
#Load WTK data
import pandas as pd

tea_prices = pd.read_csv('wtk_true_monthly_2015_2026.csv')
tea_prices.columns = tea_prices.columns.str.lower()
tea_prices['date'] = pd.to_datetime(tea_prices['date'], errors='coerce')
tea_prices = tea_prices.sort_values('date')
monthly_tea_prices = tea_prices.resample('ME', on='date').mean(numeric_only=True).reset_index()
monthly_ndvi = monthly_ndvi.reset_index()
monthly_ndvi.columns = monthly_ndvi.columns.str.lower()
monthly_ndvi['date'] = pd.to_datetime(monthly_ndvi['date'], errors='coerce')
monthly_ndvi = monthly_ndvi.sort_values('date')

data = pd.merge_asof(monthly_ndvi, monthly_tea_prices, on='date')
data.to_csv("Tea Project Data.csv")

print(data.head(5))
