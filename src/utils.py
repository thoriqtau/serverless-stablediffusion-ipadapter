import requests
import base64
import cv2
import numpy as np

from PIL import Image, ImageOps
from io import BytesIO

def crop_face(image, face, margin=0.3, output_size=(512, 512)):
    x1, y1, x2, y2 = face.bbox.astype(int)
    h, w = y2 - y1, x2 - x1
    size = int(max(h, w) * (1 + margin))

    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    half = size // 2

    x1, x2 = cx - half, cx + half
    y1, y2 = cy - half, cy + half

    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)

    cropped = image[y1:y2, x1:x2]

    resized = cv2.resize(cropped, output_size, interpolation=cv2.INTER_LANCZOS4)

    return resized

def convert_from_cv2_to_image(img: np.ndarray) -> Image:
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def convert_from_image_to_cv2(image: Image) -> np.ndarray:
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

def load_image(image_file: str):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content))
    else:
        image_bytes = base64.b64decode(image_file)
        image = Image.open(BytesIO(image_bytes))

    image = ImageOps.exif_transpose(image)
    image = image.convert('RGB')
    return image