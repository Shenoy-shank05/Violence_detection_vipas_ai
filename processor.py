import numpy as np
import tensorflow as tf
from io import BytesIO
from PIL import Image  # Ensure this import is included
import base64

def pre_process(input_image_base64):
    # Decode the base64 image input
    image_data = base64.b64decode(input_image_base64)
    
    # Open the image and convert to RGB
    original_image = Image.open(BytesIO(image_data)).convert('RGB')
    
    # Resize the image to match model input size
    image = original_image.resize((224, 224))
    
    # Convert the image to a numpy array and normalize it
    image_np = np.array(image) / 255.0  # Normalization to [0, 1]
    
    # Add a batch dimension for model input
    image_np = np.expand_dims(image_np, axis=0)  # Shape (1, 224, 224, 3)
    
    return image_np.tolist(), original_image  # Convert to list for JSON serialization

def post_process(predictions):
    boxes = predictions['detection_boxes'].numpy()
    scores = predictions['detection_scores'].numpy()
    classes = predictions['detection_classes'].numpy()
    num_detections = int(predictions['num_detections'].numpy())

    filtered_boxes = []
    filtered_scores = []
    filtered_classes = []

    for i in range(num_detections):
        if scores[i] > 0.5:  # Threshold for confidence
            filtered_boxes.append(boxes[i].tolist())  # Convert to list for JSON serialization
            filtered_scores.append(float(scores[i]))  # Convert to float for JSON serialization
            filtered_classes.append(int(classes[i]))  # Convert to int for JSON serialization

    filtered_predictions = {
        'detection_boxes': filtered_boxes,
        'detection_scores': filtered_scores,
        'detection_classes': filtered_classes,
        'num_detections': len(filtered_scores)
    }
    
    return filtered_predictions
