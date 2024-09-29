import numpy as np
import tensorflow as tf
from io import BytesIO
from PIL import Image
import base64

def pre_process(input_image_base64):
    """
    Preprocess the input image for the model.
    
    Parameters:
        input_image_base64 (str): Base64 encoded input image.
        
    Returns:
        tuple: A tuple containing:
            - List: Preprocessed image array ready for model input.
            - str: Base64 encoded original image.
    """
    # Decode the base64 image input
    image_data = base64.b64decode(input_image_base64)
    
    # Open the image and convert to RGB
    original_image = Image.open(BytesIO(image_data)).convert('RGB')
    
    # Resize the image to match model input size (128x128)
    image = original_image.resize((128, 128))
    
    # Convert the image to a numpy array and normalize it to [0, 1]
    image_np = np.array(image) / 255.0
    
    # Add a batch dimension for model input (shape: (1, 128, 128, 3))
    image_np = np.expand_dims(image_np, axis=0)

    # Convert the original image to a base64 string for output
    buffered = BytesIO()
    original_image.save(buffered, format="PNG")  # Save as PNG
    original_image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return image_np.tolist(), original_image_base64  # Convert to list for JSON serialization


def post_process(predictions):
    """
    Post-process the model predictions to extract relevant information.
    
    Parameters:
        predictions (tf.Tensor): The raw output from the model.
        
    Returns:
        dict: A dictionary containing filtered detection results.
    """
    boxes = predictions['detection_boxes'].numpy()  # Get bounding box coordinates
    scores = predictions['detection_scores'].numpy()  # Get confidence scores
    classes = predictions['detection_classes'].numpy()  # Get detected classes
    num_detections = int(predictions['num_detections'].numpy())  # Get number of detections

    # Lists to hold filtered predictions
    filtered_boxes = []
    filtered_scores = []
    filtered_classes = []

    # Filter out predictions based on score threshold
    for i in range(num_detections):
        if scores[i] > 0.5:  # Confidence threshold
            filtered_boxes.append(boxes[i].tolist())  # Convert to list for JSON serialization
            filtered_scores.append(float(scores[i]))  # Convert to float for JSON serialization
            filtered_classes.append(int(classes[i]))  # Convert to int for JSON serialization

    # Compile filtered predictions into a dictionary
    filtered_predictions = {
        'detection_boxes': filtered_boxes,
        'detection_scores': filtered_scores,
        'detection_classes': filtered_classes,
        'num_detections': len(filtered_scores)
    }
    
    return filtered_predictions
