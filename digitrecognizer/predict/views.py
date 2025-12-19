import os
import json
import base64
import uuid
import numpy as np
from io import BytesIO
from PIL import Image, ImageOps,ImageFilter
from django.http import JsonResponse
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt
from .model.model import predict
from django.shortcuts import render

def upload_image(request):
    if request.method != "POST":
        return JsonResponse({"error": "Invalid method"}, status=400)

    data = json.loads(request.body)
    image_data = data["image"].split(",")[1]

    img = Image.open(BytesIO(base64.b64decode(image_data))).convert("L")
    img=img.resize((28,28))
    img=ImageOps.grayscale(img)

    img_array = np.array(img) / 255.0
    img_array = img_array.flatten().reshape(1, -1)

    prediction = predict(img_array)

    return JsonResponse({
        "status": "success",
        "prediction": int(prediction)})

def index(request):
    return render(request,'draw.html')