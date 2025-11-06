from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# --- Everything else here is global imports ---
from PIL import Image                # 👈 must be here
import torch
from torchvision import transforms
import io
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.enums import Resampling
from shapely.geometry import box, mapping
import json
import os

from model import HybridCNNTransformer


 

app = FastAPI()

# --- Load model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_classes = 10  # EuroSAT has 10 classes
class_names = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"
]

model = HybridCNNTransformer(num_classes=num_classes)
model.load_state_dict(torch.load("models/best_model.pth", map_location=device))
model.to(device)
model.eval()

# --- Image transform ---
imagenet_mean, imagenet_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std)
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image bytes
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Preprocess
        input_tensor = transform(image).unsqueeze(0).to(device)

        # Inference
        with torch.no_grad():
            output = model(input_tensor)
            _, pred = torch.max(output, 1)
            predicted_class = class_names[pred.item()]

        return JSONResponse({
            "status": "success",
            "predicted_class": predicted_class
        })

    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

# --- Image transform ---
imagenet_mean, imagenet_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std)
])


def read_image(file_bytes: bytes) -> Image.Image:
    """Read both normal images (JPG/PNG) and GeoTIFFs."""
    try:
        return Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except Exception:
        with rasterio.MemoryFile(file_bytes) as memfile:
            with memfile.open() as src:
                array = src.read([1, 2, 3], out_shape=(3, src.height, src.width),
                                 resampling=Resampling.bilinear)
                array = np.moveaxis(array, 0, -1)
                array = np.clip(array, 0, np.percentile(array, 99))
                array = (array / array.max() * 255).astype(np.uint8)
                return Image.fromarray(array)


def split_image(image: Image.Image, tile_size=128):
    """Split image into tiles and record coordinates."""
    w, h = image.size
    patches, coords = [], []
    for top in range(0, h, tile_size):
        for left in range(0, w, tile_size):
            box_coords = (left, top, min(left + tile_size, w), min(top + tile_size, h))
            patch = image.crop(box_coords)
            patches.append(patch)
            coords.append(box_coords)
    return patches, coords


@app.post("/predict-multi")
async def predict_multi(file: UploadFile = File(...)):
    try:
        # Save temporarily
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as f:
            f.write(await file.read())

        with rasterio.open(temp_path) as src:
            width, height = src.width, src.height
            transform_affine = src.transform
            tile_size = 128

            features = []
            predictions = []

            for top in range(0, height, tile_size):
                for left in range(0, width, tile_size):
                    w = min(tile_size, width - left)
                    h = min(tile_size, height - top)
                    window = Window(left, top, w, h)
                    data = src.read([1, 2, 3], window=window, resampling=Resampling.bilinear)
                    data = np.moveaxis(data, 0, -1)

                    # Skip empty/black tiles
                    if np.all(data == 0):
                        continue

                    # Normalize and preprocess
                    data = np.clip(data, 0, np.percentile(data, 99))
                    data = (data / data.max() * 255).astype(np.uint8)

                    img = Image.fromarray(data)
                    input_tensor = transform(img).unsqueeze(0).to(device)

                    with torch.no_grad():
                        output = model(input_tensor)
                        _, pred = torch.max(output, 1)
                        cls_idx = pred.item()
                        cls_name = class_names[cls_idx]
                        predictions.append(cls_name)

                    bbox = rasterio.windows.bounds(window, transform_affine)
                    geom = box(*bbox)
                    features.append({
                        "type": "Feature",
                        "properties": {"class": cls_name},
                        "geometry": mapping(geom)
                    })

        # --- Compute class distribution ---
        total = len(predictions)
        class_counts = {cls: predictions.count(cls) for cls in class_names}
        class_distribution = {cls: round(count / total, 3) for cls, count in class_counts.items() if count > 0}
        dominant_class = max(class_distribution, key=class_distribution.get)

        # --- Save GeoJSON ---
        geojson_obj = {"type": "FeatureCollection", "features": features}
        os.makedirs("outputs", exist_ok=True)
        out_path = os.path.join("outputs", f"geo_pred_{file.filename}.geojson")
        with open(out_path, "w") as f:
            json.dump(geojson_obj, f)

        # --- Response ---
        return JSONResponse({
            "status": "success",
            "dominant_class": dominant_class,
            "class_distribution": class_distribution,
            "count": total,
            "saved_to": out_path,
            "geojson": geojson_obj
        })

    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)
    try:
        # Save temporarily
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as f:
            f.write(await file.read())

        with rasterio.open(temp_path) as src:
            width, height = src.width, src.height
            transform_affine = src.transform
            crs = src.crs
            tile_size = 128

            features = []
            predictions = []

            # Slide window through image
            for top in range(0, height, tile_size):
                for left in range(0, width, tile_size):
                    w = min(tile_size, width - left)
                    h = min(tile_size, height - top)
                    window = Window(left, top, w, h)
                    data = src.read([1, 2, 3], window=window, resampling=Resampling.bilinear)
                    data = np.moveaxis(data, 0, -1)

                    if np.all(data == 0):
                        continue

                    data = np.clip(data, 0, np.percentile(data, 99))
                    data = (data / data.max() * 255).astype(np.uint8)

                    img = Image.fromarray(data)
                    input_tensor = transform(img).unsqueeze(0).to(device)

                    with torch.no_grad():
                        output = model(input_tensor)
                        _, pred = torch.max(output, 1)
                        cls_idx = pred.item()
                        cls_name = class_names[cls_idx]
                        predictions.append(cls_name)

                    # Geographic bounds
                    bbox = rasterio.windows.bounds(window, transform_affine)
                    geom = box(*bbox)
                    features.append({
                        "type": "Feature",
                        "properties": {"class": cls_name},
                        "geometry": mapping(geom)
                    })

        # --- Compute class distribution ---
        total = len(predictions)
        class_counts = {cls: predictions.count(cls) for cls in class_names}
        class_distribution = {cls: round(count / total, 3) for cls, count in class_counts.items() if count > 0}
        dominant_class = max(class_distribution, key=class_distribution.get)

        # --- Save GeoJSON ---
        geojson_obj = {"type": "FeatureCollection", "features": features}
        os.makedirs("outputs", exist_ok=True)
        out_path = os.path.join("outputs", f"geo_pred_{file.filename}.geojson")
        with open(out_path, "w") as f:
            json.dump(geojson_obj, f)

        # --- Response ---
        return JSONResponse({
            "status": "success",
            "dominant_class": dominant_class,
            "class_distribution": class_distribution,
            "count": total,
            "saved_to": out_path,
            "geojson": geojson_obj
        })

    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)
    try:
        # Save the uploaded file temporarily
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as f:
            f.write(await file.read())

        # Open as GeoTIFF
        with rasterio.open(temp_path) as src:
            width, height = src.width, src.height
            transform_affine = src.transform
            crs = src.crs
            tile_size = 128
            features = []

            # Iterate through image by tiles
            for top in range(0, height, tile_size):
                for left in range(0, width, tile_size):
                    w = min(tile_size, width - left)
                    h = min(tile_size, height - top)
                    window = Window(left, top, w, h)
                    data = src.read([1, 2, 3], window=window, resampling=Resampling.bilinear)
                    data = np.moveaxis(data, 0, -1)

                    # Normalize and convert to image tensor
                    if np.all(data == 0):
                        continue  # skip empty tiles
                    data = np.clip(data, 0, np.percentile(data, 99))
                    data = (data / data.max() * 255).astype(np.uint8)

                    # Convert to tensor
                    # from PIL import Image
                    img = Image.fromarray(data)
                    input_tensor = transform(img).unsqueeze(0).to(device)

                    with torch.no_grad():
                        output = model(input_tensor)
                        _, pred = torch.max(output, 1)
                        cls_name = class_names[pred.item()]

                    # Compute geographic bounds
                    bbox = rasterio.windows.bounds(window, transform_affine)
                    geom = box(*bbox)
                    feature = {
                        "type": "Feature",
                        "properties": {"class": cls_name},
                        "geometry": mapping(geom)
                    }
                    features.append(feature)

        # Save to GeoJSON
        geojson_obj = {"type": "FeatureCollection", "features": features}

        os.makedirs("outputs", exist_ok=True)
        out_path = os.path.join("outputs", f"geo_pred_{file.filename}.geojson")
        with open(out_path, "w") as f:
            json.dump(geojson_obj, f)

        return JSONResponse({
            "status": "success",
            "count": len(features),
            "saved_to": out_path,
            "geojson": geojson_obj
        })

    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)