import cv2
import numpy as np
import rasterio
from rasterio.features import shapes
from skimage import measure, morphology
from shapely.geometry import shape, mapping
from shapely.ops import transform as shp_transform
import pyproj
import fiona
import json
# !!!--------------------------
# Projection system for simplification is based on EPSG:32648 (UTM zone 48N). Not EPSG:3857.
# output_geojson is projected back to EPSG:4326 (WGS84) at the end.
# !!!--------------------------


# --------------------------
# User parameters
# --------------------------
input_img = "input.png"
binary_img = "binary_map.png"         # intermediate step (optional)
bbox_geojson = "bounding_box.json"    # top-left, top-right, bottom-right, bottom-left
threshold = 100                       # threshold for red detection
simplify_tolerance_m = 2.0            # simplification tolerance (meters)
min_area_ratio = 0.0001               # proportion of total image pixels for noise removal
output_geojson = "output.geojson"     # final polygon output


# --------------------------
# 1. Load image and create binary map
# --------------------------
img = cv2.imread(input_img)  # BGR format
b, g, r = cv2.split(img)

# Binary mask: red-dominant pixels
binary_map = np.where((r > threshold) & (g < threshold) & (b < threshold), 1, 0)
cv2.imwrite(binary_img, (binary_map * 255).astype(np.uint8))  # optional debug output


# --------------------------
# 2. Load bounding box GeoJSON (EPSG:4326)
# --------------------------
with open(bbox_geojson) as f:
    bbox_data = json.load(f)

bbox_geom = shape(bbox_data)
min_lon, min_lat, max_lon, max_lat = bbox_geom.bounds


# --------------------------
# 3. Get raster dimensions and define transform
# --------------------------
with rasterio.open(binary_img) as src:
    raster = src.read(1)
height, width = raster.shape
print(f"Raster size: {width} × {height}")

from rasterio.transform import from_bounds
transform = from_bounds(
    min_lon, min_lat,  # west, south
    max_lon, max_lat,  # east, north
    width, height
)


# --------------------------
# 4. Threshold and remove noise (adaptive min area)
# --------------------------
mask = raster > 0
min_area_pixels = int(min_area_ratio * height * width)
print(f"Using min_area_pixels = {min_area_pixels}")
mask = morphology.remove_small_objects(mask.astype(bool), min_size=min_area_pixels)

# Label connected components
labels = measure.label(mask)


# --------------------------
# 5. Polygonize connected components
# --------------------------
polygons = []
for region in measure.regionprops(labels):
    coords = region.coords
    single_mask = np.zeros_like(mask, dtype=np.uint8)
    single_mask[tuple(coords.T)] = 1

    for geom, val in shapes(single_mask, mask=single_mask, transform=transform):
        if val == 1:
            poly = shape(geom)
            if poly.area > 0:
                polygons.append(poly)


# --------------------------
# 6. Simplify polygons using UTM (meters)
# --------------------------
# Determine approximate UTM zone from bounding box centroid
centroid_lon = (min_lon + max_lon) / 2
centroid_lat = (min_lat + max_lat) / 2
utm_zone = int((centroid_lon + 180) / 6) + 1
utm_crs = f"+proj=utm +zone={utm_zone} +datum=WGS84 +units=m +no_defs"

print(f"Using UTM zone {utm_zone} for simplification (tolerance = {simplify_tolerance_m} m)")

# Define coordinate transformers
to_utm = pyproj.Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True).transform
to_wgs = pyproj.Transformer.from_crs(utm_crs, "EPSG:4326", always_xy=True).transform

simplified_polygons = []
for poly in polygons:
    if not poly.is_valid or poly.area <= 0:
        continue

    poly_m = shp_transform(to_utm, poly)  # project to meters
    poly_simplified_m = poly_m.simplify(simplify_tolerance_m, preserve_topology=True)
    poly_simplified = shp_transform(to_wgs, poly_simplified_m)  # back to EPSG:4326
    simplified_polygons.append(poly_simplified)

polygons = simplified_polygons


# --------------------------
# 7. Save to GeoJSON (EPSG:4326)
# --------------------------
geojson_features = []
for idx, poly in enumerate(polygons):
    geojson_features.append({
        "type": "Feature",
        "geometry": mapping(poly),
        "properties": {"id": idx}
    })

geojson_dict = {
    "type": "FeatureCollection",
    "crs": {"type": "name", "properties": {"name": "EPSG:4326"}},
    "features": geojson_features
}

with open(output_geojson, "w") as f:
    json.dump(geojson_dict, f)

print(f"✅ Polygonized buildings saved to {output_geojson} (EPSG:4326)")
