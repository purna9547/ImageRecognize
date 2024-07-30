from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from tensorflow.keras.models import load_model
import io
import os
import cv2
import numpy as np

# Define the classes for your model predictions
classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

# Create your views here.
def home(request):
    return render(request, 'index.html')

@csrf_exempt
def image(request):
    print(os.getcwd())
    if request.method == 'POST':
        
        # Check if 'file' is in request.FILES
        if 'file' in request.FILES:
            image_file = request.FILES['file']
            image_stream = io.BytesIO(image_file.read())
            image = cv2.imdecode(np.frombuffer(image_stream.read(), np.uint8), 1)
            
            image = preprocess(image)
            model = load_model(os.path.join(os.getcwd(),'model2.keras'))
            pred = model.predict(image)
            
            pred_idx = np.argmax(pred[0])
            
            return JsonResponse({
                'object_name': classes[int(pred_idx)],
                'prediction_confidence': float(pred[0][pred_idx] * 100)
            })
        else:
            return JsonResponse({'error': 'No file uploaded'}, status=400)
    return JsonResponse({'error': 'Invalid method'}, status=405)

def preprocess(image):
    image = cv2.resize(image, (32, 32))
    image = image / 255.0
    image = image.reshape(1, 32, 32, 3)
    return image



def live(request):
    return render(request,'cam.html')